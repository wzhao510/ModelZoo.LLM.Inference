# SPDX-License-Identifier: Apache-2.0
"""Multi-node (mp backend) cluster manager.

This manager allocates *nodes* (not individual GPUs) from a cluster-config file and
starts/stops vLLM mp multi-node *headless* ranks on remote nodes via SSH.

Design notes:
- This tool assumes `vllm` can be executed directly on each node (same container/env).
- The local machine running `launch.py` is expected to be the rank0 (master) node,
  which should correspond to the first node in the cluster config file.
- For simplicity the manager assumes a fixed `gpu_per_node` (default: 8),
  consistent with the original RayClusterManager.
"""

from __future__ import annotations

import shlex
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import paramiko
import regex as re
from pprint import pprint
import os 

AuthType = Literal["password", "key"]

@dataclass
class SSHConfig:
    hostname: str
    port: int = 22
    user: str = "root"
    auth_type: AuthType = "password"  # "password" or "key"
    password: Optional[str] = None
    private_key: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "SSHConfig":
        raw = str(d.get("auth_type", "password")).lower().strip()
        if raw in ("password", "pass", "passwd"):
            auth_type: AuthType = "password"
        elif raw in ("key", "sshkey", "pkey", "private_key"):
            auth_type = "key"
        else:
            raise ValueError(f"Unsupported auth_type: {d.get('auth_type')}")

        return SSHConfig(
            hostname=d["hostname"],
            port=int(d.get("port", 22)),
            user=d.get("user", "root"),
            auth_type=auth_type,
            password=d.get("password"),
            private_key=d.get("private_key") or d.get("keyfile") or d.get("ssh_key"),
    )


@dataclass
class ClusterNode:
    ssh: SSHConfig
    nic: str = "eth0"
    extra_env: Optional[dict[str, Any]] = None  # from cluster.yaml per-node extra_env

    @staticmethod
    def from_dict(d: dict) -> "ClusterNode":
        ssh_cfg = SSHConfig.from_dict(d["ssh"])
        nic = d.get("nic") or (d.get("ray", {}) or {}).get("nic") or "eth0"
        extra_env = d.get("extra_env") or None
        if extra_env is not None and not isinstance(extra_env, dict):
            raise ValueError("cluster-config extra_env must be a dict mapping")
        return ClusterNode(ssh=ssh_cfg, nic=nic, extra_env=extra_env)


def wrap_command_with_env(command: list[str] | str, env_dict: dict[str, Any]) -> str:
    """bash -lc 'export ... && <cmd>'"""
    if isinstance(command, list):
        cmd_str = " ".join(shlex.quote(x) for x in command)
    else:
        cmd_str = command

    exports = []
    for k, v in (env_dict or {}).items():
        if v is None:
            continue
        exports.append(f"export {k}={shlex.quote(str(v))}")

    inner = " && ".join(exports + [cmd_str]) if exports else cmd_str
    return f"bash -lc {shlex.quote(inner)}"


def remote_command(ssh_cfg: SSHConfig, command: str, timeout: int = 60) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs = dict(
            hostname=ssh_cfg.hostname,
            port=ssh_cfg.port,
            username=ssh_cfg.user,
            timeout=timeout,
        )

        if ssh_cfg.auth_type == "key":
            if not ssh_cfg.private_key:
                raise ValueError("auth_type=key but private_key is not set in cluster config.")
            connect_kwargs.update(
                key_filename=ssh_cfg.private_key,
                allow_agent=True,
                look_for_keys=False,
            )
        else:
            if ssh_cfg.password is None:
                raise ValueError("auth_type=password but password is not set in cluster config.")
            connect_kwargs.update(
                password=ssh_cfg.password,
                allow_agent=False,
                look_for_keys=False,
            )

        client.connect(**connect_kwargs)
        _stdin, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(
                f"Remote command failed (exit={exit_status}) on {ssh_cfg.hostname}:{ssh_cfg.port}\n"
                f"CMD: {command}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
            )
        return out.strip()
    finally:
        try:
            client.close()
        except Exception:
            pass


class MPClusterManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cluster_config: dict | list):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.global_mutex = threading.Lock()
        self.occupied_nodes: set[int] = set()

        self.all_nodes: list[ClusterNode] = self._init_node(cluster_config)
        pprint(self.all_nodes)

        self.gpu_per_node = 8
        self.all_gpu_nums = len(self.all_nodes) * self.gpu_per_node

        self._remote_vllm_pids: dict[int, int] = {}

    def _init_node(self, node_config: list | dict) -> ClusterNode | list[ClusterNode]:
        if isinstance(node_config, list):
            return [self._init_node(item) for item in node_config]
        return ClusterNode.from_dict(node_config)

    # ---------- helpers ----------
    def get_node_hostname(self, node_idx: int) -> str:
        return self.all_nodes[node_idx].ssh.hostname

    def get_base_env(self, node_idx: int) -> dict[str, Any]:
        """Centralized base env: only MCCL (no NCCL)."""
        node = self.all_nodes[node_idx]
        env: dict[str, Any] = {
            "GLOO_SOCKET_IFNAME": node.nic,
            "MCCL_SOCKET_IFNAME": node.nic,
            "MACA_PATH": "/opt/maca",
            "MACA_DIRECT_DISPATH": "1",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        }
        if node.extra_env:
            env.update(node.extra_env)
        return env

    # backward compatibility (if some code still calls it)
    def get_nic_env(self, node_idx: int) -> dict[str, Any]:
        return self.get_base_env(node_idx)

    def get_gpu_process_pid(self, *_args, **_kwargs) -> list[int]:
        return []

    def _parse_pid(self, output: str) -> int:
        if not output:
            raise RuntimeError("Empty output while starting remote vLLM.")
        nums = re.findall(r"\b(\d+)\b", output)
        if not nums:
            raise RuntimeError(f"Failed to parse PID from output: {output}")
        return int(nums[-1])

    # ---------- public ----------
    def get_free_nodes(self) -> list[int]:
        return [i for i in range(len(self.all_nodes)) if i not in self.occupied_nodes]

    def allocate(self, num_required: int) -> list[int]:
        if num_required > self.all_gpu_nums:
            raise ValueError("Requested more GPUs than available on the system.")

        needed_nodes = (num_required + self.gpu_per_node - 1) // self.gpu_per_node

        with self.global_mutex:
            free_nodes = self.get_free_nodes()
            if len(free_nodes) < needed_nodes:
                return []
            allocated = free_nodes[:needed_nodes]
            self.occupied_nodes.update(allocated)
            return allocated

    def start_headless_rank(self, node_idx: int, cmd: list[str], env: dict, log_path: str) -> int:
        node = self.all_nodes[node_idx]
        full_env = {**self.get_base_env(node_idx), **(env or {})}
        remote_inner = wrap_command_with_env(["exec", *cmd], full_env)  
        remote_cmd = (
            f"nohup setsid {remote_inner} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        )
        print(
            f"node_idx [{node_idx}] execute command: {remote_cmd}"
        )
        output = remote_command(node.ssh, remote_cmd)
        pid = self._parse_pid(output)
        self._remote_vllm_pids[node_idx] = pid
        return pid

    def stop_headless_rank(self, node_idx: int, master_port: int | None = None):
        node = self.all_nodes[node_idx]
        pid = self._remote_vllm_pids.get(node_idx)
        if pid is None:
            return
        kill_group_cmd = (
            "bash -lc " + shlex.quote(
                f"kill -TERM -{pid} 2>/dev/null || true; "
                f"sleep 3; "
                f"kill -KILL -{pid} 2>/dev/null || true"
            )
        )
        try:
            remote_command(node.ssh, kill_group_cmd)
        except Exception as e:
            print(f"[MPClusterManager] killpg failed on node {node_idx}: {e}")

        self._remote_vllm_pids.pop(node_idx, None)

    def release(self, related_nodes: list[int]):
        with self.global_mutex:
            for i in related_nodes[1:]:
                try:
                    self.stop_headless_rank(i)
                except Exception as e:
                    print(f"[MPClusterManager] Failed stopping rank on node {i}: {e}")
            for i in related_nodes:
                self.occupied_nodes.discard(i)
