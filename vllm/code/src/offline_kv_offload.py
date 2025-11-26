import pytest
import os
import subprocess
import time
import requests
import socket
import textwrap
from pathlib import Path
import contextlib 


try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("错误：请先安装 'openai' 和 'transformers' 库 (pip install openai transformers)")
    exit(1)


# --- 测试配置常量 ---
# vLLM 模型路径
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
# KVCache 测试服务的主机和端口
VLLM_HOST = "localhost"
VLLM_PORT = 8300  # 两个测试将依次使用同一个端口
# vLLM 实例的最大模型长度
MAX_MODEL_LEN = 16384
# 创建一个接近最大长度的长上下文
TEST_PROMPT_TOKENS = 15000
# 使用哪个 GPU
CUDA_DEVICE_ID = "0"


# --- 上下文管理器：自动化管理 vLLM 卸载服务 ---

@contextlib.contextmanager
def vllm_offload_service(offload_type, tmp_path_factory):
    """
    一个上下文管理器，负责：
    1. 根据 offload_type ("cpu" 或 "disk") 创建 KVCache 配置文件。
    2. 设置必要的环境变量 (LMCACHE_USE_EXPERIMENTAL)。
    3. 启动一个 vLLM 服务，并配置为 "kv_both" 角色。
    4. 等待 vLLM 服务端口就绪。
    5. 将服务信息 (端口, 日志文件) yield 给测试函数。
    6. 在所有测试结束后，终止 vLLM 服务进程。
    """
    
    # 1. 在临时目录中创建配置文件和日志文件
    test_run_path = tmp_path_factory.mktemp(f"kv_offload_{offload_type}_test")
    config_file = test_run_path / "lmcache_config.yaml"
    log_file = test_run_path / "vllm_server.log"
    
    # 如果是磁盘模式，在临时目录中创建一个子目录用于存放缓存文件
    disk_cache_path = test_run_path / "disk_cache_storage"

    print(f"\n[调试信息] KVCache {offload_type} 卸载测试，日志文件位于: {log_file}")
    
    # 根据参数 ("cpu" or "disk") 生成 LMCache 配置文件
    config_content = ""
    if offload_type == "cpu":
        # CPU 卸载配置
        config_content = textwrap.dedent(f"""
            chunk_size: 256
            local_cpu: true
            max_local_cpu_size: 5.0
            local_disk: None
            max_local_disk_size: 0
        """)
    elif offload_type == "disk":
        # 磁盘卸载配置
        config_content = textwrap.dedent(f"""
            chunk_size: 256
            local_cpu: false
            max_local_cpu_size: 5.0 
            local_disk: "file://{disk_cache_path}"
            max_local_disk_size: 5.0
        """)
        
    config_file.write_text(config_content)
    print(f"  > LMCache 配置文件 ({config_file}) 内容:\n{config_content}")

    # 设置环境变量
    original_env = os.environ.copy()
    env_vars_to_set = {
        "LMCACHE_USE_EXPERIMENTAL": "True",
        "LMCACHE_CONFIG_FILE": str(config_file),
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICE_ID
    }
    os.environ.update(env_vars_to_set)

    process = None
    log_file_handle = None
    try:
        # 启动 vLLM 服务
        print("\n--- 正在启动 vLLM KVCache 卸载服务 ---")
        log_file_handle = open(log_file, "w")
        
        vllm_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(VLLM_PORT),
            "--disable-log-requests",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
        ]
        
        print(f"  > vLLM: {' '.join(vllm_cmd)}")
        process = subprocess.Popen(vllm_cmd, env=os.environ.copy(), stdout=log_file_handle, stderr=subprocess.STDOUT)

        # 等待 vLLM 服务加载模型并开放端口
        print("  > 等待 vLLM 服务加载模型 (可能需要几分钟)...")
        _wait_for_port(VLLM_HOST, VLLM_PORT, timeout=180)
        print("  > vLLM 服务已就绪。")

        # 将控制权和信息交给 "with" 语句块
        yield {
            "port": VLLM_PORT,
            "host": VLLM_HOST,
            "log_file": log_file,
            "type": offload_type
        }

    finally:
        # 测试结束后，清理所有资源
        print(f"\n--- 正在关闭 vLLM ({offload_type} 卸载) 服务 ---")
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  > 进程 {process.pid} 被强制终止。")
        
        if log_file_handle:
            log_file_handle.close()
            
        print("--- 服务已关闭 ---")
        # 恢复环境变量
        os.environ.clear()
        os.environ.update(original_env)

# --- 辅助函数  ---

def _wait_for_port(host, port, timeout=180):
    """在指定时间内持续检查端口是否开放"""
    start_time = time.monotonic()
    print(f"  > 正在等待端口 {host}:{port} ...")
    while True:
        try:
            with socket.create_connection((host, port), timeout=10):
                print(f"  > 端口 {port} 已开放。")
                return
        except (ConnectionRefusedError, OSError):
            if time. monotonic() - start_time >= timeout:
                raise TimeoutError(f"等待端口 {port} 超时 ({timeout}s)。服务可能启动失败。")
            time.sleep(1)

def _get_long_prompt():
    """使用 tokenizer 创建一个指定 token 数量的超长提示"""
    print(f"  > 正在从 {VLLM_MODEL_PATH} 加载 tokenizer 以生成长提示...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH)
    except Exception as e:
        print(f"  > 警告：无法加载本地 tokenizer。将使用 'gpt2' 替代。错误: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (TEST_PROMPT_TOKENS - 10) # 留点余量
    long_context = tokenizer.decode(long_context_tokens)
    question = "\n\nSummarize the text above in exactly one sentence."
    prompt = long_context + question
    
    final_tokens = tokenizer.encode(prompt)
    if len(final_tokens) > MAX_MODEL_LEN:
        print(f"  > 警告：生成的提示 ({len(final_tokens)} tokens) 超过了最大长度 ({MAX_MODEL_LEN})。")
        prompt = tokenizer.decode(final_tokens[:MAX_MODEL_LEN-50])

    print(f"  > 已生成长提示，Token 数量: {len(tokenizer.encode(prompt))}")
    return prompt, tokenizer

def query_and_measure_ttft(client, model_id, prompt):
    """发送流式请求并测量第一个 token 的返回时间 (TTFT)"""
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
                    print(f"  > 收到第一个 token: '{chunk.choices[0].delta.content}'", end="", flush=True)
                else:
                    print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n  > 流式响应结束。")
        if first_token_time is None:
            raise RuntimeError("流式响应中没有收到任何内容。")
        return first_token_time - start_time
    except Exception as e:
        print(f"\n  > 查询时发生错误: {e}")
        return 9999.0

def _run_ttft_test(service_info):
    """
    一个可重用的函数，封装了完整的 TTFT 测试逻辑。
    1. 连接客户端
    2. 获取长提示
    3. 执行冷热查询
    4. 打印结果并断言
    """
    base_url = f"http://{service_info['host']}:{service_info['port']}/v1"
    test_type = service_info['type']
    
    print(f"\n--- [测试] 正在测试 KVCache {test_type} 卸载 ---")
    print(f"  > 服务地址: {base_url}")
    print(f"  > 服务日志: {service_info['log_file']}")

    # 准备 OpenAI 客户端
    client = OpenAI(api_key="dummy-key", base_url=base_url)
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"  > 成功连接到 vLLM 服务, 模型 ID: {model_id}")
    except Exception as e:
        pytest.fail(f"无法连接到 vLLM 服务: {e}。请检查服务日志: {service_info['log_file']}")

    # 准备长提示语
    prompt, _ = _get_long_prompt()

    # --- 冷缓存查询 ---
    print("\n--- [测试] 正在进行第一次查询 (冷缓存)... ---")
    cold_ttft = query_and_measure_ttft(client, model_id, prompt)
    print(f"\n✅ 冷缓存 TTFT: {cold_ttft:.3f} 秒")
    
    assert cold_ttft > 0, "冷缓存查询太快了，可能 KVCache 未生效或 prompt 太短。"

    # --- 热缓存查询 ---
    time.sleep(1) # 确保 LMCache 已完成异步存储
    print("\n--- [测试] 正在进行第二次查询 (热缓存)... ---")
    warm_ttft = query_and_measure_ttft(client, model_id, prompt)
    print(f"\n✅ 热缓存 TTFT: {warm_ttft:.3f} 秒")

    # --- 验证结果 ---
    improvement = cold_ttft - warm_ttft
    speedup_factor = cold_ttft / warm_ttft
    print("\n" + "="*30)
    print(f" KVCache {test_type} 卸载测试结果:")
    print(f"   冷缓存 TTFT: {cold_ttft:.3f} 秒")
    print(f"   热缓存 TTFT: {warm_ttft:.3f} 秒")
    print(f"   TTFT 提升: {improvement:.3f} 秒 ({speedup_factor:.1f}x 更快)")
    print("="*30)

    # 热缓存必须明显快于冷缓存
    assert warm_ttft < (cold_ttft / 2), f"热缓存 ({warm_ttft:.3f}s) 不够快，没有达到冷缓存 ({cold_ttft:.3f}s) 的 1/2。"
    
    print(f"✅ KVCache {test_type} 卸载测试通过!")


# --- 两个独立的测试用例 ---

def test_01_kvcache_cpu_offload(tmp_path_factory):
    """
    测试用例 1: 验证 CPU RAM 卸载。
    
    tmp_path_factory 是一个 pytest 内置 fixture，用于创建临时目录。
    """
    print("\n" + "#"*70)
    print("### 开始测试用例 1: CPU KVCache 卸载 ###")
    print("#"*70)
    
    # "with" 语句会启动服务，并在代码块结束时自动关闭服务
    with vllm_offload_service("cpu", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)
        
    print("\n### 测试用例 1: CPU KVCache 卸载 完成 ###")


def test_02_kvcache_disk_offload(tmp_path_factory):
    """
    测试用例 2: 验证 磁盘 (Disk) 卸载。
    (Pytest 会在 test_01 之后运行这个)
    """
    print("\n" + "#"*70)
    print("### 开始测试用例 2: 磁盘 KVCache 卸载 ###")
    print("#"*70)
    
    # "with" 语句会启动一个 新的服务，并在代码块结束时自动关闭服务
    with vllm_offload_service("disk", tmp_path_factory) as service_info:
        _run_ttft_test(service_info)

    print("\n### 测试用例 2: 磁盘 KVCache 卸载 完成 ###")
