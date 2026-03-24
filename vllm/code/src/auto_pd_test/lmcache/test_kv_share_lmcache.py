import pytest
import os
import subprocess
import time
import socket
import textwrap
from pathlib import Path
import contextlib
import json

try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("错误：请先安装 'openai' 和 'transformers' 库 (pip install openai transformers)")
    exit(1)

# --- 测试配置常量 ---
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "localhost"
# 实例1 和 实例2 的端口
VLLM_PORT_1 = 8000
VLLM_PORT_2 = 8001
# 实例1 和 实例2 使用的 GPU
GPU_ID_1 = "3"
GPU_ID_2 = "4"

# --- 中心化 LMCache 服务器端口 ---
CENTRALIZED_SERVER_PORT = 65432

# --- P2P v2 (Controller) 新增常量 ---
CONTROLLER_PORT = 9000
CONTROLLER_PULL_PORT = 8300
CONTROLLER_REPLY_PORT = 8400

# P2P v2 vLLM 1 的端口
P2P_INIT_PORT_1 = 8200
P2P_LOOKUP_PORT_1 = 8201
P2P_WORKER_PORT_1 = 8500

# P2P v2 vLLM 2 的端口
P2P_INIT_PORT_2 = 8202
P2P_LOOKUP_PORT_2 = 8203
P2P_WORKER_PORT_2 = 8501


# --- 上下文管理器 1: 中心化 KVCache 共享服务 ---
@contextlib.contextmanager
def centralized_sharing_service(tmp_path_factory):
    """
    一个上下文管理器，负责启动和管理“中心化共享”所需的所有服务。
    - 1x LMCache Server
    - 2x vLLM Instances
    """
    test_run_path = tmp_path_factory.mktemp("centralized_sharing_test")
    config_file = test_run_path / "lmcache_config.yaml"
    log_paths = {
        "lmcache_server": test_run_path / "lmcache_server.log",
        "vllm1": test_run_path / "vllm1.log",
        "vllm2": test_run_path / "vllm2.log",
    }
    print(f"\n[调试信息] 中心化共享测试，日志文件位于: {test_run_path}")

    # 1. 创建共享的 LMCache 配置文件
    config_content = textwrap.dedent(f"""
        chunk_size: 256
        local_cpu: true
        remote_url: "lm://{VLLM_HOST}:{CENTRALIZED_SERVER_PORT}"
        remote_serde: "cachegen"
    """)
    config_file.write_text(config_content)

    processes = []
    log_handles = {}
    original_env = os.environ.copy()
    try:
        # 2. 启动 LMCache 中心化服务器
        print("\n--- 正在启动 LMCache 中心化服务器 ---")
        log_handles["lmcache_server"] = open(log_paths["lmcache_server"], "w")
        lmcache_cmd = ["lmcache_server", VLLM_HOST, str(CENTRALIZED_SERVER_PORT)]

        lmcache_env = original_env.copy()
        lmcache_env["PYTHONHASHSEED"] = "1289"
        lmcache_env["LMCACHE_LOG_LEVEL"] = "DEBUG"
        lmcache_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

        p_lmcache = subprocess.Popen(lmcache_cmd, stdout=log_handles["lmcache_server"], stderr=subprocess.STDOUT)
        processes.append(p_lmcache)
        _wait_for_port(VLLM_HOST, CENTRALIZED_SERVER_PORT)

        # 3. 准备 vLLM 环境变量
        vllm_env = original_env.copy()
        vllm_env["LMCACHE_CONFIG_FILE"] = str(config_file)
        vllm_env["PYTHONHASHSEED"] = "1289"
        vllm_env["LMCACHE_LOG_LEVEL"] = "DEBUG"
        vllm_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        # 4. 启动两个 vLLM 实例
        print("\n--- 正在启动两个 vLLM 实例 ---")
        log_handles["vllm1"] = open(log_paths["vllm1"], "w")
        vllm_env1 = vllm_env.copy(); vllm_env1["CUDA_VISIBLE_DEVICES"] = GPU_ID_1
        vllm_cmd1 = _get_vllm_cmd(VLLM_PORT_1)
        p_vllm1 = subprocess.Popen(vllm_cmd1, env=vllm_env1, stdout=log_handles["vllm1"], stderr=subprocess.STDOUT)
        processes.append(p_vllm1)

        log_handles["vllm2"] = open(log_paths["vllm2"], "w")
        vllm_env2 = vllm_env.copy(); vllm_env2["CUDA_VISIBLE_DEVICES"] = GPU_ID_2
        vllm_cmd2 = _get_vllm_cmd(VLLM_PORT_2)
        p_vllm2 = subprocess.Popen(vllm_cmd2, env=vllm_env2, stdout=log_handles["vllm2"], stderr=subprocess.STDOUT)
        processes.append(p_vllm2)

        # 5. 等待所有服务就绪
        print("\n--- 等待所有服务端口开放 ---")
        _wait_for_port(VLLM_HOST, VLLM_PORT_1)
        _wait_for_port(VLLM_HOST, VLLM_PORT_2)
        print("✅ 所有服务已成功启动并准备就绪!")

        # 6. 将控制权交给测试函数
        yield {
            "vllm1_port": VLLM_PORT_1, "vllm2_port": VLLM_PORT_2,
            "host": VLLM_HOST, "type": "中心化"
        }

    finally:
        # 7. 测试结束后，清理所有资源
        print("\n--- 正在关闭所有服务 ---")
        _terminate_processes(processes)
        for f in log_handles.values(): f.close()
        os.environ.clear(); os.environ.update(original_env)

# --- 上下文管理器 2: P2P KVCache 共享服务 ---
@contextlib.contextmanager
def p2p_sharing_service(tmp_path_factory):
    """
    (已更新) 一个上下文管理器，负责启动和管理“P2P v2 共享”所需的所有服务。
    - 1x LMCache Controller (替换了 Redis)
    - 2x vLLM Instances (使用 NIXL 和新配置)
    """
    test_run_path = tmp_path_factory.mktemp("p2p_v2_sharing_test")
    config1_file = test_run_path / "lmcache_config1.yaml"
    config2_file = test_run_path / "lmcache_config2.yaml"
    log_paths = {
        "controller": test_run_path / "controller.log",
        "vllm1": test_run_path / "vllm1_p2p.log",
        "vllm2": test_run_path / "vllm2_p2p.log",
    }
    print(f"\n[调试信息] P2P v2 共享测试，日志文件位于: {test_run_path}")
    
    # --- 1. 生成新的 P2P 配置文件 (vLLM 1) ---
    config1_content = textwrap.dedent(f"""
        chunk_size: 256
        local_cpu: True
        max_local_cpu_size: 9
        enable_async_loading: True
        
        # P2P configurations
        enable_p2p: True
        p2p_host: "{VLLM_HOST}"
        p2p_init_ports: {P2P_INIT_PORT_1}
        p2p_lookup_ports: {P2P_LOOKUP_PORT_1}
        transfer_channel: "nixl"

        # Controller configurations
        enable_controller: True
        lmcache_instance_id: "lmcache_instance_1"
        controller_pull_url: "{VLLM_HOST}:{CONTROLLER_PULL_PORT}"
        controller_reply_url: "{VLLM_HOST}:{CONTROLLER_REPLY_PORT}"
        lmcache_worker_ports: {P2P_WORKER_PORT_1}

        extra_config:
          lookup_backoff_time: 0.001
    """)
    config1_file.write_text(config1_content)

    # --- 2. 生成新的 P2P 配置文件 (vLLM 2) ---
    config2_content = textwrap.dedent(f"""
        chunk_size: 256
        local_cpu: True
        max_local_cpu_size: 9
        enable_async_loading: True

        # P2P configurations
        enable_p2p: True
        p2p_host: "{VLLM_HOST}"
        p2p_init_ports: {P2P_INIT_PORT_2}
        p2p_lookup_ports: {P2P_LOOKUP_PORT_2}
        transfer_channel: "nixl"

        # Controller configurations
        enable_controller: True
        lmcache_instance_id: "lmcache_instance_2"
        controller_pull_url: "{VLLM_HOST}:{CONTROLLER_PULL_PORT}"
        controller_reply_url: "{VLLM_HOST}:{CONTROLLER_REPLY_PORT}"
        lmcache_worker_ports: {P2P_WORKER_PORT_2}

        extra_config:
          lookup_backoff_time: 0.001
    """)
    config2_file.write_text(config2_content)

    processes = []
    log_handles = {}
    original_env = os.environ.copy()
    
    try:
        # --- 3. 启动 LMCache Controller (替换 Redis) ---
        print("\n--- 正在启动 LMCache Controller ---")
        log_handles["controller"] = open(log_paths["controller"], "w")
        
        # 构造 --monitor-ports 的 JSON 字符串参数
        monitor_ports_config = json.dumps({
            "pull": CONTROLLER_PULL_PORT, 
            "reply": CONTROLLER_REPLY_PORT
        })
        
        controller_cmd = [
            "lmcache_controller",
            "--host", VLLM_HOST,
            "--port", str(CONTROLLER_PORT),
            "--monitor-ports", monitor_ports_config
        ]
        
        controller_env = original_env.copy()
        controller_env["PYTHONHASHSEED"] = "1289" # 保持哈希种子一致
        controller_env["LMCACHE_LOG_LEVEL"] = "DEBUG"
        controller_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        
        p_controller = subprocess.Popen(controller_cmd, env=controller_env, stdout=log_handles["controller"], stderr=subprocess.STDOUT)
        processes.append(p_controller)
        
        # 等待 Controller 的所有端口
        _wait_for_port(VLLM_HOST, CONTROLLER_PORT)
        _wait_for_port(VLLM_HOST, CONTROLLER_PULL_PORT)
        _wait_for_port(VLLM_HOST, CONTROLLER_REPLY_PORT)
        
        # --- 4. 准备 vLLM 环境变量 (UCX_TLS) ---
        vllm_base_env = original_env.copy()
        vllm_base_env["PYTHONHASHSEED"] = "1289"
        vllm_base_env["UCX_TLS"] = "rc" # 根据文档，NIXL 传输需要
        vllm_base_env["LMCACHE_LOG_LEVEL"] = "DEBUG"
        vllm_base_env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

        # --- 5. 启动两个 vLLM 实例 (P2P 模式) ---
        print("\n--- 正在启动两个 vLLM 实例 (P2P v2 模式) ---")
        
        log_handles["vllm1"] = open(log_paths["vllm1"], "w")
        vllm_env1 = vllm_base_env.copy()
        vllm_env1["LMCACHE_CONFIG_FILE"] = str(config1_file)
        vllm_env1["CUDA_VISIBLE_DEVICES"] = GPU_ID_1
        vllm_cmd1 = _get_vllm_cmd(VLLM_PORT_1)
        p_vllm1 = subprocess.Popen(vllm_cmd1, env=vllm_env1, stdout=log_handles["vllm1"], stderr=subprocess.STDOUT)
        processes.append(p_vllm1)

        log_handles["vllm2"] = open(log_paths["vllm2"], "w")
        vllm_env2 = vllm_base_env.copy()
        vllm_env2["LMCACHE_CONFIG_FILE"] = str(config2_file)
        vllm_env2["CUDA_VISIBLE_DEVICES"] = GPU_ID_2
        vllm_cmd2 = _get_vllm_cmd(VLLM_PORT_2)
        p_vllm2 = subprocess.Popen(vllm_cmd2, env=vllm_env2, stdout=log_handles["vllm2"], stderr=subprocess.STDOUT)
        processes.append(p_vllm2)

        # --- 6. 等待 vLLM 服务就绪 ---
        print("\n--- 等待所有服务端口开放 ---")
        _wait_for_port(VLLM_HOST, VLLM_PORT_1)
        _wait_for_port(VLLM_HOST, VLLM_PORT_2)
        print("✅ 所有服务已成功启动并准备就绪!")
        
        yield {
            "vllm1_port": VLLM_PORT_1, "vllm2_port": VLLM_PORT_2,
            "host": VLLM_HOST, "type": "P2P v2" # 更新测试类型名称
        }

    finally:
        # --- 7. 清理逻辑 ---
        print("\n--- 正在关闭所有服务 (包括 lmcache_controller) ---")
        _terminate_processes(processes) # 会终止所有进程
        for f in log_handles.values(): f.close()
        os.environ.clear(); os.environ.update(original_env)

# --- 辅助函数 ---
def _get_vllm_cmd(port):
    """生成 vLLM 服务的启动命令"""
    return [
        "vllm", "serve", VLLM_MODEL_PATH,
        "--port", str(port),
        # "--disable-log-requests",
        "--gpu-memory-utilization", "0.8",
        # kv_role: kv_both 是 P2P 模式所必需的
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
    ]


def _terminate_processes(processes):
    """终止给定的子进程列表"""
    for p in processes: p.terminate()
    for p in processes:
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill(); print(f"  > 进程 {p.pid} 被强制终止。")

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

def _get_long_prompt(num_tokens=3072):
    """创建一个较长的提示语来有效测试 KVCache"""
    print(f"  > 正在生成约 {num_tokens} tokens 的长提示...")
    # 使用一个简单的重复文本来构造长上下文
    base_text = "This is a long context to test the Key-Value Cache sharing feature in LMCache with vLLM. Repeating this sentence multiple times generates a significant number of tokens. "
    repetitions = (num_tokens * 4) // len(base_text)  # 估算重复次数
    long_context = base_text * repetitions
    question = "\n\nBased on the text above, what is being tested?"
    return long_context + question

def query_completions_and_measure_ttft(client, model_id, prompt):
    """使用 /v1/completions 接口发送请求并测量 TTFT"""
    start_time = time.perf_counter()
    try:
        response = client.completions.create(
            model=model_id,
            prompt=prompt,
            temperature=0.0,
            max_tokens=128,
        )
        first_token_time = time.perf_counter() # 对于非流式，响应返回时间即为 TTFT
        
        text = response.choices[0].text.strip().replace('\n', ' ')
        print(f"  > 收到响应: '{text[:50]}...'")
        return first_token_time - start_time
    except Exception as e:
        print(f"\n  > 查询时发生错误: {e}")
        return 9999.0

def _run_sharing_test(service_info):
    """可重用的函数，封装了完整的共享测试逻辑"""
    host = service_info['host']
    port1 = service_info['vllm1_port']
    port2 = service_info['vllm2_port']
    test_type = service_info['type']
    
    # 准备两个 OpenAI 客户端
    client1 = OpenAI(api_key="dummy-key", base_url=f"http://{host}:{port1}/v1")
    client2 = OpenAI(api_key="dummy-key", base_url=f"http://{host}:{port2}/v1")
    
    try:
        model_id = client1.models.list().data[0].id
        print(f"  > 成功连接到 vLLM 服务, 模型 ID: {model_id}")
    except Exception as e:
        pytest.fail(f"无法连接到 vLLM 服务: {e}。")

    prompt = _get_long_prompt()

    # --- 1. 冷缓存查询 (实例1) ---
    print(f"\n--- [测试] 正在向实例1 (端口 {port1}) 发送请求 (冷缓存)... ---")
    cold_ttft = query_completions_and_measure_ttft(client1, model_id, prompt)
    print(f"\n✅ 实例1 (冷) TTFT: {cold_ttft:.3f} 秒")

    print("\n--- [测试] 等待 2 秒，确保 KVCache 已在对等方之间同步... ---")
    time.sleep(2)
    
    # --- 2. 热缓存查询 (实例2) ---
    print(f"\n--- [测试] 正在向实例2 (端口 {port2}) 发送请求 (热缓存)... ---")
    warm_ttft = query_completions_and_measure_ttft(client2, model_id, prompt)
    print(f"\n✅ 实例2 (热) TTFT: {warm_ttft:.3f} 秒")

    # --- 3. 验证结果 ---
    improvement = cold_ttft - warm_ttft
    speedup_factor = cold_ttft / warm_ttft
    print("\n" + "="*30)
    print(f" KVCache {test_type} 共享测试结果:")
    print(f"   实例1 (冷) TTFT: {cold_ttft:.3f} 秒")
    print(f"   实例2 (热) TTFT: {warm_ttft:.3f} 秒")
    print(f"   TTFT 提升: {improvement:.3f} 秒 ({speedup_factor:.1f}x 更快)")
    print("="*30)

    
    assert warm_ttft < (cold_ttft * 5 / 5), f"热缓存 ({warm_ttft:.3f}s) 不够快，没有超过冷缓存 ({cold_ttft:.3f}s) 的 时间。"
    print(f"✅ KVCache {test_type} 共享测试通过!")



def test_01_centralized_sharing(tmp_path_factory):
    """测试用例 1: 验证中心化 KVCache 共享"""
    print("\n" + "#"*70)
    print("### 开始测试用例 1: 中心化 KVCache 共享 ###")
    print("#"*70)
    with centralized_sharing_service(tmp_path_factory) as service_info:
        _run_sharing_test(service_info)
    print("\n### 测试用例 1: 中心化 KVCache 共享 完成 ###")


def test_02_p2p_sharing(tmp_path_factory):
    """测试用例 2: 验证 P2P KVCache 共享 (已更新)"""
    print("\n" + "#"*70)
    print("### 开始测试用例 2: P2P KVCache 共享 (v2) ###")
    print("#"*70)
    with p2p_sharing_service(tmp_path_factory) as service_info:
        _run_sharing_test(service_info)
    print("\n### 测试用例 2: P2P KVCache 共享 (v2) 完成 ###")
