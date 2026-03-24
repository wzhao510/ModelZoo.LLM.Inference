import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-8B/"
TEST_DIR = Path(__file__).resolve().parent
PROXY_SCRIPT_PATH = TEST_DIR / "disagg_proxy_mooncake_server.py"

HOST = "127.0.0.1"
PROXY_PORT = 9487
PREFILLER_PORT = 7100
DECODER1_PORT = 7200
DECODER2_PORT = 7201

MOONCAKE_PORT_PREFILLER = 8998
MOONCAKE_PORT_DECODER1 = 8999
MOONCAKE_PORT_DECODER2 = 9000

GPU_MEMORY_UTILIZATION = 0.85


def _dump_process_logs(log_paths, log_handles=None, tail_lines=200):
    print("\n" + "=" * 80)
    print("STARTUP DEBUG LOG DUMP")
    print("=" * 80)
    if log_handles:
        for handle in log_handles.values():
            try:
                handle.flush()
            except Exception:
                pass

    for name, log_path in log_paths.items():
        print(f"\n--- {name}: {log_path} ---")
        try:
            with open(log_path, "r", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("(log file not found)")
            continue

        if not lines:
            print("(log file is empty)")
            continue

        print("".join(lines[-tail_lines:]).rstrip())

    print("\n" + "=" * 80)


def _wait_for_port(host, port, timeout=300, processes=None, log_paths=None, log_handles=None, stage="startup"):
    start = time.time()
    while time.time() - start < timeout:
        if processes:
            for name, proc in processes.items():
                retcode = proc.poll()
                if retcode is not None:
                    print(f"[FATAL] Process {name} exited unexpectedly during {stage} with code {retcode}.")
                    if log_paths:
                        _dump_process_logs(log_paths, log_handles)
                    raise RuntimeError(f"Process {name} exited unexpectedly during {stage} with code {retcode}.")

        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
            if int(time.time()) % 10 == 0:
                print(f"Waiting for {host}:{port}...")

    if log_paths:
        print(f"[FATAL] Timeout waiting for {host}:{port} during {stage}.")
        _dump_process_logs(log_paths, log_handles)
    raise TimeoutError(f"Port {port} not ready in {timeout}s")


@pytest.fixture(scope="module")
def cluster_1p2d_mooncake(tmp_path_factory):
    print("\n[Setup] Starting Mooncake 1P2D Cluster (TP=2)...")
    run_dir = tmp_path_factory.mktemp("mooncake_1p2d_tp2")
    print(f"[Setup] Logs will be saved to: {run_dir}")

    processes = {}
    logs = {}
    log_paths = {
        "prefiller": run_dir / "prefiller.log",
        "decoder1": run_dir / "decoder1.log",
        "decoder2": run_dir / "decoder2.log",
        "proxy": run_dir / "proxy.log",
    }

    base_env = os.environ.copy()
    base_env["MC_FORCE_RDMA"] = "1"
    base_env["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
    base_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    try:
        print(f"-> Launching Prefiller (Port {PREFILLER_PORT}, Mooncake {MOONCAKE_PORT_PREFILLER})...")
        logs["pre"] = open(log_paths["prefiller"], "w")
        p_env = base_env.copy()
        p_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        p_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(PREFILLER_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen",
            "--kv-transfer-config", json.dumps({
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
            }),
        ]
        processes["prefiller"] = subprocess.Popen(p_cmd, env=p_env, stdout=logs["pre"], stderr=subprocess.STDOUT)

        print(f"-> Launching Decoder 1 (Port {DECODER1_PORT}, Mooncake {MOONCAKE_PORT_DECODER1})...")
        logs["d1"] = open(log_paths["decoder1"], "w")
        d1_env = base_env.copy()
        d1_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        d1_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER1_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen",
            "--kv-transfer-config", json.dumps({
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
            }),
        ]
        processes["decoder1"] = subprocess.Popen(d1_cmd, env=d1_env, stdout=logs["d1"], stderr=subprocess.STDOUT)

        print(f"-> Launching Decoder 2 (Port {DECODER2_PORT}, Mooncake {MOONCAKE_PORT_DECODER2})...")
        logs["d2"] = open(log_paths["decoder2"], "w")
        d2_env = base_env.copy()
        d2_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        d2_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER2_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen",
            "--kv-transfer-config", json.dumps({
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
            }),
        ]
        processes["decoder2"] = subprocess.Popen(d2_cmd, env=d2_env, stdout=logs["d2"], stderr=subprocess.STDOUT)

        print("Waiting for vLLM instances to initialize...")
        _wait_for_port(HOST, PREFILLER_PORT, processes=processes, log_paths=log_paths, log_handles=logs, stage="prefiller startup")
        _wait_for_port(HOST, DECODER1_PORT, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder1 startup")
        _wait_for_port(HOST, DECODER2_PORT, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder2 startup")

        print(f"-> Launching Proxy on Port {PROXY_PORT}...")
        logs["proxy"] = open(log_paths["proxy"], "w")
        proxy_cmd = [
            "python3", str(PROXY_SCRIPT_PATH),
            "--port", str(PROXY_PORT),
            "--host", HOST,
            "--prefill", f"http://{HOST}:{PREFILLER_PORT}", f"{MOONCAKE_PORT_PREFILLER}",
            "--decod", f"http://{HOST}:{DECODER1_PORT}",
            "--decod", f"http://{HOST}:{DECODER2_PORT}", 
        ]
        processes["proxy"] = subprocess.Popen(proxy_cmd, stdout=logs["proxy"], stderr=subprocess.STDOUT)
        _wait_for_port(HOST, PROXY_PORT, 60, processes=processes, log_paths=log_paths, log_handles=logs, stage="proxy startup")

        print("[Setup] Cluster is READY!")
        yield

    finally:
        print("\n[Teardown] Cleaning up processes...")
        for p in processes.values():
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass

        time.sleep(3)

        for p in processes.values():
            try:
                if p.poll() is None:
                    print(f"Force killing process {p.pid}")
                    p.kill()
            except Exception:
                pass

        for f in logs.values():
            f.close()
        print("[Teardown] Done.")


def send_chat_request(request_id, prompt):
    print(f"\n[Req #{request_id}] Sending: '{prompt}'")
    url = f"http://{HOST}:{PROXY_PORT}/v1/chat/completions"

    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": True,
    }

    collected_text = []
    received_done = False

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as resp:
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
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            collected_text.append(content)
                    except Exception:
                        pass
    except Exception as e:
        return False, str(e)

    full_response = "".join(collected_text)
    print(f"[Req #{request_id} Result] {full_response}")

    if not received_done:
        return False, "Stream interrupted (No [DONE])"
    return True, full_response


def test_mooncake_pd_round_robin(cluster_1p2d_mooncake):
    test_cases = [
        {"prompt": "10 + 10 = ?", "expect": "20"},
        {"prompt": "1 + 2 = ?", "expect": "3"},
    ]

    failures = []

    for i, case in enumerate(test_cases):
        req_id = i + 1
        success, result = send_chat_request(req_id, case["prompt"])

        if not success:
            failures.append(f"Req #{req_id} Failed: {result}")
        elif case["expect"].lower() not in result.lower():
            failures.append(f"Req #{req_id} Content Error. Expected '{case['expect']}', Got: '{result}'")
        else:
            print(f"Req #{req_id} Passed")

        time.sleep(2.0)

    if failures:
        pytest.fail("\n".join(failures))

