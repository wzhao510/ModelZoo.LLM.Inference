import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


TEST_DIR = Path(__file__).resolve().parent
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-8B/"
PROXY_SCRIPT_PATH = TEST_DIR / "disagg_proxy_p2p_nccl_xpyd.py"

HOST = "127.0.0.1"
ENTRY_POINT_PORT = 10001
PROXY_INTERNAL_PORT = 30001

PREFILL_CONFIG = [
    {"name": "pre-1", "gpus": "0,1", "port": 8100, "kv_port": 21000},
]
DECODE_CONFIG = [
    {"name": "dec-1", "gpus": "2,3", "port": 8200, "kv_port": 22000},
    {"name": "dec-2", "gpus": "4,5", "port": 8201, "kv_port": 22010},
]


def dump_logs(logs_map, tail_lines=200):
    print("\n" + "=" * 80)
    print("STARTUP DEBUG LOG DUMP")
    print("=" * 80)
    for name, filepath in logs_map.items():
        path_obj = Path(filepath)
        print(f"\n--- {name}: {path_obj} ---")
        if not path_obj.exists():
            print("(log file not found)")
            continue
        try:
            with open(path_obj, "r", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading log file: {e}")
            continue

        if not lines:
            print("(log file is empty)")
            continue

        print("".join(lines[-tail_lines:]).rstrip())
    print("\n" + "=" * 80 + "\n")


def wait_for_cluster_ready(processes, ports_to_check, log_files_map, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        for name, proc in processes.items():
            ret_code = proc.poll()
            if ret_code is not None:
                print(f"\n[FATAL] Process {name} exited unexpectedly with code {ret_code}!")
                dump_logs(log_files_map)
                raise RuntimeError(f"Process {name} died during startup.")

        all_ready = True
        not_ready_list = []
        for port in ports_to_check:
            try:
                with socket.create_connection((HOST, port), timeout=0.1):
                    pass
            except (socket.timeout, ConnectionRefusedError, OSError):
                all_ready = False
                not_ready_list.append(port)

        if all_ready:
            print(f"\n[Success] All ports {ports_to_check} are ready!")
            return

        elapsed = int(time.time() - start_time)
        if elapsed % 5 == 0:
            print(f"Waiting for ports {not_ready_list}... ({elapsed}/{timeout}s)")
        time.sleep(1)

    print(f"\n[FATAL] Timeout waiting for ports: {not_ready_list}")
    dump_logs(log_files_map)
    raise TimeoutError(f"Cluster failed to start in {timeout}s")


@pytest.fixture(scope="module")
def cluster_p2p_nccl_tp2(tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("p2p_run_logs")
    print(f"\n[Info] Test Artifacts Directory: {log_dir}")
    print(f"[Info] Logs will be saved to: {log_dir}")
    print(f"\n[Setup] Starting P2P NCCL Cluster (1P2D, TP=2)...")

    procs = {}
    log_paths = {}
    all_ports = [ENTRY_POINT_PORT]

    base_env = os.environ.copy()
    base_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    base_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    try:
        print(f"-> Launching Proxy (Public: {ENTRY_POINT_PORT})...")
        log_paths["proxy"] = log_dir / "proxy.log"
        proxy_out = open(log_paths["proxy"], "w")
        proxy_cmd = ["python3", str(PROXY_SCRIPT_PATH), "--port", str(ENTRY_POINT_PORT)]
        proxy_env = base_env.copy()
        proxy_env["PROXY_PORT"] = str(PROXY_INTERNAL_PORT)
        procs["proxy"] = subprocess.Popen(proxy_cmd, env=proxy_env, stdout=proxy_out, stderr=subprocess.STDOUT)
        time.sleep(1)

        for conf in PREFILL_CONFIG:
            print(f"-> Launching {conf['name']} (GPUs {conf['gpus']}, Port {conf['port']})...")
            all_ports.append(conf["port"])
            log_paths[conf["name"]] = log_dir / f"{conf['name']}.log"
            f_out = open(log_paths[conf["name"]], "w")
            p_env = base_env.copy()
            p_env["CUDA_VISIBLE_DEVICES"] = conf["gpus"]
            kv_config = {
                "kv_connector": "P2pNcclConnector",
                "kv_role": "kv_producer",
                "kv_buffer_size": "1e1",
                "kv_port": str(conf["kv_port"]),
                "kv_connector_extra_config": {
                    "proxy_ip": "0.0.0.0",
                    "proxy_port": str(PROXY_INTERNAL_PORT),
                    "http_port": str(conf["port"]),
                    "send_type": "PUT_ASYNC",
                    "nccl_num_channels": "16",
                },
            }
            cmd = [
                "vllm", "serve", VLLM_MODEL_PATH,
                "--port", str(conf["port"]),
                "--gpu-memory-utilization", "0.9",
                "--tensor-parallel-size", "2",
                "--max-model-len", "4096",
                "--enforce-eager",
                "--trust-remote-code",
                "--kv-transfer-config", json.dumps(kv_config),
            ]
            procs[conf["name"]] = subprocess.Popen(cmd, env=p_env, stdout=f_out, stderr=subprocess.STDOUT)

        for conf in DECODE_CONFIG:
            print(f"-> Launching {conf['name']} (GPUs {conf['gpus']}, Port {conf['port']})...")
            all_ports.append(conf["port"])
            log_paths[conf["name"]] = log_dir / f"{conf['name']}.log"
            f_out = open(log_paths[conf["name"]], "w")
            d_env = base_env.copy()
            d_env["CUDA_VISIBLE_DEVICES"] = conf["gpus"]
            kv_config = {
                "kv_connector": "P2pNcclConnector",
                "kv_role": "kv_consumer",
                "kv_buffer_size": "4e9",
                "kv_port": str(conf["kv_port"]),
                "kv_connector_extra_config": {
                    "proxy_ip": "0.0.0.0",
                    "proxy_port": str(PROXY_INTERNAL_PORT),
                    "http_port": str(conf["port"]),
                    "send_type": "PUT_ASYNC",
                    "nccl_num_channels": "16",
                },
            }
            cmd = [
                "vllm", "serve", VLLM_MODEL_PATH,
                "--port", str(conf["port"]),
                "--gpu-memory-utilization", "0.8",
                "--tensor-parallel-size", "2",
                "--max-model-len", "4096",
                "--enforce-eager",
                "--trust-remote-code",
                "--kv-transfer-config", json.dumps(kv_config),
            ]
            procs[conf["name"]] = subprocess.Popen(cmd, env=d_env, stdout=f_out, stderr=subprocess.STDOUT)

        print("Waiting for cluster to initialize...")
        wait_for_cluster_ready(procs, all_ports, log_paths)
        yield

    finally:
        print("\n[Teardown] Cleaning up...")
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        time.sleep(2)

        for p in procs.values():
            if p.poll() is None:
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass

        os.system(f"pkill -9 -f {PROXY_SCRIPT_PATH.name}")
        print(f"[Teardown] Logs are available at: {log_dir}")


def send_chat_request(request_id, prompt):
    print(f"\n[Req #{request_id}] Sending: '{prompt}'")
    url = f"http://{HOST}:{ENTRY_POINT_PORT}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0,
        "stream": True,
    }

    collected_text = []
    received_done = False
    try:
        with requests.post(url, json=payload, stream=True, timeout=30) as resp:
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}: {resp.text}"

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    json_str = decoded[5:].strip()
                    if json_str == "[DONE]":
                        received_done = True
                        break
                    try:
                        data = json.loads(json_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                collected_text.append(content)
                    except Exception:
                        pass
    except Exception as e:
        return False, str(e)

    full = "".join(collected_text)
    print(f"[Req #{request_id} Result] {full}")
    return received_done, full


def test_p2p_nccl_tp2_chat(cluster_p2p_nccl_tp2):
    test_cases = [
        {"prompt": "Say Hello", "expect": "Hello"},
        {"prompt": "1+1=?", "expect": "2"},
    ]

    failures = []
    for i, case in enumerate(test_cases):
        success, res = send_chat_request(i, case["prompt"])
        if not success or case["expect"].lower() not in res.lower():
            failures.append(f"Req {i} Failed: {res}")

    if failures:
        pytest.fail("\n".join(failures))

