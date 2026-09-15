import contextlib
import json
import os
import socket
import subprocess
import time
import textwrap

import pytest

try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("Please install 'openai' and 'transformers' first.")
    raise SystemExit(1)

# ==============================================================================
# 全局实际配置参数
# ==============================================================================
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8300
MOONCAKE_MASTER_PORT = 50051
MAX_MODEL_LEN = 16384
TEST_PROMPT_TOKENS = 15000
CUDA_DEVICE_ID = "0"
# ==============================================================================


def _dump_file(log_file, tail_lines=100):
    print("\n" + "=" * 80)
    print(f"DEBUG LOG DUMP: {log_file}")
    try:
        with open(log_file, "r", errors="ignore") as f:
            lines = f.readlines()
            print("".join(lines[-tail_lines:]).rstrip() if lines else "(empty)")
    except FileNotFoundError:
        print("(log file not found)")
    print("=" * 80)


def _wait_for_port(host, port, timeout=120, process=None, stage="startup"):
    start_time = time.monotonic()
    print(f"Waiting for port {host}:{port} ({stage})...")
    while True:
        if process and process.poll() is not None:
            raise RuntimeError(f"[{stage}] Process exited with code {process.poll()}")
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"Port {port} is ready.")
                return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() - start_time >= timeout:
                raise TimeoutError(f"Timed out waiting for port {port}.")
            time.sleep(1)


@contextlib.contextmanager
def mooncake_infrastructure(mode, tmp_path):
    """
    实际管理 Mooncake 底层基础设施的上下文管理器。
    包含拉起 mooncake_master，以及在 standalone 模式下拉起 mooncake_client。
    """
    master_proc = None
    client_proc = None
    master_log = tmp_path / "mooncake_master.log"
    client_log = tmp_path / "mooncake_client.log"
    
    try:
        # 1. 启动 mooncake_master
        master_cmd = ["mooncake_master", "--port", str(MOONCAKE_MASTER_PORT)]
        if mode == "disk":
            master_cmd.append("--enable_offload=true")
            
        print(f"\n--- Starting mooncake_master: {' '.join(master_cmd)} ---")
        master_proc = subprocess.Popen(master_cmd, stdout=open(master_log, "w"), stderr=subprocess.STDOUT)
        _wait_for_port("127.0.0.1", MOONCAKE_MASTER_PORT, timeout=30, process=master_proc, stage="mooncake_master")

        # 2. 如果是磁盘模式，启动独立的 mooncake_client 作为存储节点
        if mode == "disk":
            ssd_path = tmp_path / "ssd_storage"
            ssd_path.mkdir(exist_ok=True)
            
            client_env = os.environ.copy()
            client_env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] = str(ssd_path)
            
            client_cmd = ["mooncake_client", "--enable_offload=true"]
            print(f"--- Starting mooncake_client (Store Owner): {' '.join(client_cmd)} ---")
            print(f"SSD Path: {ssd_path}")
            
            client_proc = subprocess.Popen(client_cmd, env=client_env, stdout=open(client_log, "w"), stderr=subprocess.STDOUT)
            # 给予 client 几秒钟的初始化与 master 握手的时间
            time.sleep(3) 

        yield
        
    finally:
        print("\n--- Tearing down Mooncake infrastructure ---")
        for proc in [client_proc, master_proc]:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


@contextlib.contextmanager
def vllm_mooncake_service(test_mode, tmp_path_factory):
    """
    配置并启动使用 MooncakeStoreConnector 的 vLLM 服务。
    """
    test_run_path = tmp_path_factory.mktemp(f"mooncake_{test_mode}_test")
    config_file = test_run_path / "mooncake_config.json"
    log_file = test_run_path / "vllm_server.log"

    # 生成实际的 JSON 配置
    # 注意: 0.3.12 要求 local_hostname / metadata_server / master_server_address 都非空
    mooncake_config = {
        "local_hostname": "127.0.0.1",
        "metadata_server": f"127.0.0.1:{MOONCAKE_MASTER_PORT}",
        "master_server_address": f"127.0.0.1:{MOONCAKE_MASTER_PORT}",
        "protocol": "rdma",
        "device_name": "mlx5_0"
    }

    if test_mode == "cpu_embedded":
        mooncake_config.update({
            "mode": "embedded",
            "global_segment_size": "10GB", # 实际分配 10GB CPU 内存
            "local_buffer_size": "2GB",
            "enable_offload": False
        })
    elif test_mode == "disk_standalone":
        mooncake_config.update({
            "mode": "standalone-store",
            "global_segment_size": 0, # Requester 不贡献内存
            "local_buffer_size": "2GB",
            "enable_offload": True
        })

    config_file.write_text(json.dumps(mooncake_config, indent=2))
    print(f"\n[Config] Mooncake JSON ({config_file}):\n{config_file.read_text()}")

    # 配置基于文档要求的环境变量
    # 注意: Mooncake 0.3.12 的 load_from_env 会优先用环境变量，必须显式设置
    # 避免回退到默认 P2PHANDSHAKE 模式（随机端口发现，单实例会失败）
    master_addr = f"127.0.0.1:{MOONCAKE_MASTER_PORT}"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICE_ID,
        "MOONCAKE_CONFIG_PATH": str(config_file),
        "MOONCAKE_MASTER": master_addr,
        "MOONCAKE_TE_META_DATA_SERVER": master_addr,
        "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
        "MOONCAKE_PROTOCOL": "rdma",
        "MOONCAKE_DEVICE": "mlx5_0",
        "MC_FORCE_RDMA": "1",
        "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "PYTHONHASHSEED": "0", # 保证多进程哈希一致性
        "VLLM_MOONCAKE_STORE_TIER_LOG": "1" # 开启可观测性日志
    })

    vllm_cmd = [
        "vllm", "serve", VLLM_MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--kv-transfer-config", '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    ]

    process = None
    try:
        # 使用基础设施管理器拉起 Master 和 Client
        with mooncake_infrastructure(mode="disk" if "disk" in test_mode else "cpu", tmp_path=test_run_path):
            print(f"\n--- Starting vLLM (Mooncake {test_mode}) service ---")
            print(f"Command: {' '.join(vllm_cmd)}")
            
            process = subprocess.Popen(vllm_cmd, env=env, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
            _wait_for_port(VLLM_HOST, VLLM_PORT, timeout=240, process=process, stage="vLLM Load Model")
            print("vLLM service is ready.")

            yield {
                "port": VLLM_PORT,
                "host": VLLM_HOST,
                "log_file": log_file,
                "type": test_mode,
            }

    except Exception as e:
        _dump_file(log_file)
        raise e
    finally:
        print(f"\n--- Stopping vLLM ---")
        if process:
            process.terminate()
            process.wait()


def _get_long_prompt():
    print(f"Loading tokenizer from {VLLM_MODEL_PATH} to build a long prompt...")
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH, trust_remote_code=True)
    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (TEST_PROMPT_TOKENS - 10)
    prompt = tokenizer.decode(long_context_tokens) + "\n\nSummarize exactly one sentence."
    return prompt


def query_and_measure_ttft(client, model_id, prompt):
    start_time = time.perf_counter()
    first_token_time = None
    generated_text = ""  # 新增：用于保存生成的完整文本
    
    try:
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            temperature=0.0,
            max_tokens=20,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                generated_text += content  # 新增：拼接文本
                
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    print(f"First token: '{content}'", end="", flush=True)
                else:
                    print(content, end="", flush=True)
        print()
        
        # 新增：同时返回 TTFT 时间和生成的文本
        return first_token_time - start_time, generated_text
        
    except Exception as e:
        print(f"\nQuery failed: {e}")
        return 9999.0, ""


def _run_ttft_test(service_info):
    base_url = f"http://{service_info['host']}:{service_info['port']}/v1"
    client = OpenAI(api_key="dummy-key", base_url=base_url)
    model_id = client.models.list().data[0].id
    prompt = _get_long_prompt()

    print("\n--- Cold Cache Query (Storing to Mooncake) ---")
    # 修改：接收 TTFT 和输出文本
    cold_ttft, cold_output = query_and_measure_ttft(client, model_id, prompt)
    print(f"Cold TTFT: {cold_ttft:.3f}s")

    time.sleep(3) # 给 Mooncake 异步传输引擎一点时间将 Cache 刷入共享池或磁盘

    print("\n--- Warm Cache Query (Loading from Mooncake) ---")
    # 修改：接收 TTFT 和输出文本
    warm_ttft, warm_output = query_and_measure_ttft(client, model_id, prompt)
    print(f"Warm TTFT: {warm_ttft:.3f}s")

    # 1. 判断性能
    speedup_factor = cold_ttft / warm_ttft
    print(f"\n[Result] TTFT Speedup: {speedup_factor:.1f}x faster")
    assert warm_ttft < cold_ttft * 0.7, f"Warm cache did not provide significant speedup. Cold: {cold_ttft:.3f}s, Warm: {warm_ttft:.3f}s"

    # 2. 判断正确性（新增）
    print(f"[Result] Output validation...")
    print(f"Cold Output: {repr(cold_output)}")
    print(f"Warm Output: {repr(warm_output)}")
    
    # 因为 temperature=0.0，冷热启动生成的文本必须完全一致
    assert cold_output == warm_output, "正确性测试失败：Warm Cache 的输出与 Cold Cache 不一致，底层 KV Cache 数据可能已损坏！"
    
    # 可选：确保输出不是空的（防止完全没有生成内容）
    assert len(cold_output.strip()) > 0, "正确性测试失败：模型未能生成任何有效输出。"


# ==============================================================================
# PyTest 测试用例执行入口
# ==============================================================================

def test_01_mooncake_cpu_embedded_offload(tmp_path_factory):
    """测试场景一：单节点内嵌模式下的 CPU 卸载"""
    with vllm_mooncake_service("cpu_embedded", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)


def test_02_mooncake_disk_standalone_offload(tmp_path_factory):
    """测试场景二：独立存储节点模式下的磁盘 (SSD) 卸载"""
    with vllm_mooncake_service("disk_standalone", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)