import contextlib
import json
import os
import socket
import subprocess
import textwrap
import time

import pytest

try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("Please install 'openai' and 'transformers' first.")
    raise SystemExit(1)

# ==============================================================================
# 全局配置参数
# ==============================================================================
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "localhost"
VLLM_PORT = 8300
MAX_MODEL_LEN = 16384
TEST_PROMPT_TOKENS = 15000
CUDA_DEVICE_ID = "0"
# ==============================================================================


def _dump_file(log_file, tail_lines=200):
    print("\n" + "=" * 80)
    print(f"DEBUG LOG DUMP: {log_file}")
    print("=" * 80)
    try:
        with open(log_file, "r", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("(log file not found)")
        print("=" * 80)
        return

    if not lines:
        print("(log file is empty)")
    else:
        print("".join(lines[-tail_lines:]).rstrip())
    print("=" * 80)


def _wait_for_port(host, port, timeout=180, process=None, log_file=None, stage="startup"):
    start_time = time.monotonic()
    print(f"Waiting for port {host}:{port} ...")
    while True:
        if process is not None:
            retcode = process.poll()
            if retcode is not None:
                print(f"[FATAL] Process exited unexpectedly during {stage} with code {retcode}.")
                if log_file:
                    _dump_file(log_file)
                raise RuntimeError(f"Process exited unexpectedly during {stage} with code {retcode}.")
        try:
            with socket.create_connection((host, port), timeout=10):
                print(f"Port {port} is ready.")
                return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() - start_time >= timeout:
                print(f"[FATAL] Timeout waiting for {host}:{port} during {stage}.")
                if log_file:
                    _dump_file(log_file)
                raise TimeoutError(f"Timed out waiting for port {port} ({timeout}s).")
            time.sleep(1)


@contextlib.contextmanager
def vllm_native_offload_service(offload_type, tmp_path_factory):
    """
    配置并启动 vLLM 原生 KV Cache 卸载服务。
    支持三种模式: native_cpu, tiering_cpu_disk, simple_cpu
    """
    test_run_path = tmp_path_factory.mktemp(f"kv_offload_{offload_type}_test")
    log_file = test_run_path / "vllm_server.log"
    
    # 为磁盘卸载准备具体的文件系统目录
    disk_cache_path = test_run_path / "disk_cache"
    disk_cache_path.mkdir(exist_ok=True)

    print(f"\n[Debug] Native KVCache {offload_type} offload test log: {log_file}")

    original_env = os.environ.copy()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_ID

    # 基础命令，所有原生卸载都必须开启 prefix caching
    vllm_cmd = [
        "vllm", "serve", VLLM_MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-prefix-caching" 
    ]

    # 根据具体方法注入配置
    if offload_type == "native_cpu":
        # 方法一：使用快捷参数 --kv-offloading-size
        vllm_cmd.extend(["--kv-offloading-size", "5"]) # 5 GiB CPU Offload
        
    elif offload_type == "tiering_cpu_disk":
        # 方法二：CPU + 磁盘分级卸载，使用完整的 JSON 配置
        transfer_config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 5368709120,
                "eviction_policy": "lru",
                "secondary_tiers": [
                    {"type": "fs_python", "root_dir": str(disk_cache_path)}
                ]
            }
        }
        # subprocess 可以直接接收序列化后的 JSON 字符串
        vllm_cmd.extend(["--kv-transfer-config", json.dumps(transfer_config)])
        
    elif offload_type == "simple_cpu":
        # 方法三：通过环境变量切换到 Simple 实现
        env["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "1"
        vllm_cmd.extend(["--kv-offloading-size", "5"])
        
    else:
        raise ValueError(f"Unsupported offload type: {offload_type}")

    process = None
    log_file_handle = None
    try:
        print(f"\n--- Starting vLLM ({offload_type}) service ---")
        log_file_handle = open(log_file, "w")
        
        print(f"vLLM command: {' '.join(vllm_cmd)}")
        if offload_type == "simple_cpu":
            print("Environment included: VLLM_USE_SIMPLE_KV_OFFLOAD=1")

        process = subprocess.Popen(vllm_cmd, env=env, stdout=log_file_handle, stderr=subprocess.STDOUT)

        print("Waiting for vLLM service to load the model...")
        _wait_for_port(VLLM_HOST, VLLM_PORT, timeout=180, process=process, log_file=log_file, stage=f"{offload_type} startup")
        print("vLLM service is ready.")

        yield {
            "port": VLLM_PORT,
            "host": VLLM_HOST,
            "log_file": log_file,
            "type": offload_type,
        }

    finally:
        print(f"\n--- Stopping vLLM ({offload_type}) service ---")
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"Process {process.pid} was force-killed.")

        if log_file_handle:
            log_file_handle.close()

        os.environ.clear()
        os.environ.update(original_env)


def _get_long_prompt():
    print(f"Loading tokenizer from {VLLM_MODEL_PATH} to build a long prompt...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load local tokenizer, fallback to gpt2. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (TEST_PROMPT_TOKENS - 10)
    long_context = tokenizer.decode(long_context_tokens)
    question = "\n\nSummarize the text above in exactly one sentence."
    prompt = long_context + question

    final_tokens = tokenizer.encode(prompt)
    if len(final_tokens) > MAX_MODEL_LEN:
        print(f"Warning: prompt too long ({len(final_tokens)}), truncating to fit {MAX_MODEL_LEN}.")
        prompt = tokenizer.decode(final_tokens[: MAX_MODEL_LEN - 50])

    print(f"Generated prompt tokens: {len(tokenizer.encode(prompt))}")
    return prompt, tokenizer


def query_and_measure_ttft(client, model_id, prompt):
    start_time = time.perf_counter()
    first_token_time = None
    try:
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            temperature=0.0,
            max_tokens=50,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    print(f"First token: '{chunk.choices[0].delta.content}'", end="", flush=True)
                else:
                    print(chunk.choices[0].delta.content, end="", flush=True)
        print("\nStreaming response finished.")
        if first_token_time is None:
            raise RuntimeError("No token content received from the stream.")
        return first_token_time - start_time
    except Exception as e:
        print(f"\nQuery failed: {e}")
        return 9999.0


def _run_ttft_test(service_info):
    base_url = f"http://{service_info['host']}:{service_info['port']}/v1"
    test_type = service_info["type"]
    print(f"\n--- Testing KVCache {test_type} offload ---")
    print(f"Service URL: {base_url}")
    print(f"Service log: {service_info['log_file']}")

    client = OpenAI(api_key="dummy-key", base_url=base_url)
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"Connected to vLLM service, model ID: {model_id}")
    except Exception as e:
        _dump_file(service_info["log_file"])
        pytest.fail(f"Cannot connect to vLLM service: {e}. Check log: {service_info['log_file']}")

    prompt, _ = _get_long_prompt()

    print("\n--- Cold cache query ---")
    cold_ttft = query_and_measure_ttft(client, model_id, prompt)
    print(f"\nCold cache TTFT: {cold_ttft:.3f}s")
    assert cold_ttft > 0, "Cold cache query was unexpectedly too fast."

    # 等待后台 offload 完成
    time.sleep(2)
    
    print("\n--- Warm cache query ---")
    warm_ttft = query_and_measure_ttft(client, model_id, prompt)
    print(f"\nWarm cache TTFT: {warm_ttft:.3f}s")

    improvement = cold_ttft - warm_ttft
    speedup_factor = cold_ttft / warm_ttft
    print("\n" + "=" * 30)
    print(f"KVCache {test_type} offload result:")
    print(f"  Cold cache TTFT: {cold_ttft:.3f}s")
    print(f"  Warm cache TTFT: {warm_ttft:.3f}s")
    print(f"  TTFT improvement: {improvement:.3f}s ({speedup_factor:.1f}x faster)")
    print("=" * 30)

    assert warm_ttft < (cold_ttft / 2), f"Warm cache ({warm_ttft:.3f}s) did not improve enough over cold cache ({cold_ttft:.3f}s)."


# ==============================================================================
# PyTest 测试用例
# ==============================================================================

def test_01_native_cpu_offload(tmp_path_factory):
    """测试方法一：通过 --kv-offloading-size 启用的 native CPU 卸载"""
    with vllm_native_offload_service("native_cpu", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)

def test_02_tiering_cpu_disk_offload(tmp_path_factory):
    """测试方法二：OffloadingConnector + TieringOffloadingSpec (CPU+磁盘分级)"""
    with vllm_native_offload_service("tiering_cpu_disk", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)

# 需要依赖cuda_python 我们目前并不支持该种方式 
# def test_03_simple_cpu_offload(tmp_path_factory):
#     """测试方法三：通过 VLLM_USE_SIMPLE_KV_OFFLOAD 启用的简单 CPU 卸载"""
#     with vllm_native_offload_service("simple_cpu", tmp_path_factory) as service_info:
#         _run_ttft_test(service_info)