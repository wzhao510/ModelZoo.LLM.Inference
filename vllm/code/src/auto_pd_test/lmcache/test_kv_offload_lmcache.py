import contextlib
import os
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

from common import (
    assert_cache_speedup,
    dump_file as _dump_file,
    popen_own_group,
    query_and_measure_ttft,
    terminate_process_group,
    wait_for_free_gpu_memory,
    wait_for_port as _wait_for_port,
)


VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "localhost"
VLLM_PORT = 8300
MAX_MODEL_LEN = 16384
TEST_PROMPT_TOKENS = 15000
CUDA_DEVICE_ID = "0"


@contextlib.contextmanager
def vllm_offload_service(offload_type, tmp_path_factory):
    test_run_path = tmp_path_factory.mktemp(f"kv_offload_{offload_type}_test")
    config_file = test_run_path / "lmcache_config.yaml"
    log_file = test_run_path / "vllm_server.log"
    disk_cache_path = test_run_path / "disk_cache_storage"

    print(f"\n[Debug] KVCache {offload_type} offload test log: {log_file}")

    if offload_type == "cpu":
        config_content = textwrap.dedent("""
            chunk_size: 256
            local_cpu: true
            max_local_cpu_size: 5.0
            local_disk: None
            max_local_disk_size: 0
        """)
    elif offload_type == "disk":
        config_content = textwrap.dedent(f"""
            chunk_size: 256
            local_cpu: false
            max_local_cpu_size: 5.0
            local_disk: "file://{disk_cache_path}"
            max_local_disk_size: 5.0
        """)
    else:
        raise ValueError(f"Unsupported offload type: {offload_type}")

    config_file.write_text(config_content)
    print(f"LMCache config file ({config_file}):\n{config_content}")

    original_env = os.environ.copy()
    os.environ.update({
        "LMCACHE_USE_EXPERIMENTAL": "True",
        "LMCACHE_CONFIG_FILE": str(config_file),
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICE_ID,
    })

    process = None
    log_file_handle = None
    try:
        print("\n--- Starting vLLM KVCache offload service ---")
        log_file_handle = open(log_file, "w")
        vllm_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        ]
        print(f"vLLM command: {' '.join(vllm_cmd)}")
        process = popen_own_group(vllm_cmd, env=os.environ.copy(), stdout=log_file_handle, stderr=subprocess.STDOUT)

        print("Waiting for vLLM service to load the model...")
        _wait_for_port(VLLM_HOST, VLLM_PORT, timeout=180, process=process, log_paths=log_file, stage=f"{offload_type} offload service startup")
        print("vLLM service is ready.")

        yield {
            "port": VLLM_PORT,
            "host": VLLM_HOST,
            "log_file": log_file,
            "type": offload_type,
        }

    finally:
        print(f"\n--- Stopping vLLM ({offload_type} offload) service ---")
        terminate_process_group(process, f"vLLM ({offload_type} offload)")

        if log_file_handle:
            log_file_handle.close()

        os.environ.clear()
        os.environ.update(original_env)

        if process is not None:
            wait_for_free_gpu_memory(CUDA_DEVICE_ID)


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
    cold_ttft, cold_text = query_and_measure_ttft(client, model_id, prompt, max_tokens=50)

    time.sleep(1)
    print("\n--- Warm cache query ---")
    warm_ttft, warm_text = query_and_measure_ttft(client, model_id, prompt, max_tokens=50)

    assert_cache_speedup(f"KVCache {test_type} offload", cold_ttft, warm_ttft, cold_text, warm_text)


def test_01_kvcache_cpu_offload(tmp_path_factory):
    with vllm_offload_service("cpu", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)


def test_02_kvcache_disk_offload(tmp_path_factory):
    with vllm_offload_service("disk", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)

