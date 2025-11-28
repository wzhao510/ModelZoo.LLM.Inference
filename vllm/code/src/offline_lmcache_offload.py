import pytest
import os
import subprocess
import time
import requests
import socket
import textwrap
from pathlib import Path
import contextlib 
import argparse
import sys
import tempfile
import shutil
import datetime


try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("错误：请先安装 'openai' 和 'transformers' 库 (pip install openai transformers)")
    exit(1)


# --- 测试配置参数 (通过命令行参数设置) ---
def parse_args():
    parser = argparse.ArgumentParser(description="vLLM KVCache 卸载测试脚本")
    parser.add_argument("--model", type=str, required=True,
                       help="vLLM 模型路径")
    parser.add_argument("--host", type=str, default="localhost",
                       help="KVCache 测试服务的主机")
    parser.add_argument("--port", type=int, default=8300,
                       help="KVCache 测试服务的端口")
    parser.add_argument("--max-model-len", type=int, default=16384,
                       help="vLLM 实例的最大模型长度")
    parser.add_argument("--test-prompt-tokens", type=int, default=15000,
                       help="长上下文的 token 数量")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1, help="tp")
    parser.add_argument("--test-cpu", action="store_true", default=True,
                       help="测试CPU offload")
    parser.add_argument("--test-disk", action="store_true", default=True,
                       help="测试磁盘卸载")
    parser.add_argument("--log-dir", type=str, default="./vllm_test_logs",
                       help="Server日志文件保存目录")
    
    return parser.parse_args()


# --- 上下文管理器：自动化管理 vLLM 卸载服务 ---

@contextlib.contextmanager
def vllm_offload_service(offload_type, test_run_path, args):
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
    config_file = test_run_path / "lmcache_config.yaml"
    
    # 创建专门的日志目录
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    server_log_file = log_dir / f"vllm_server_{offload_type}_{timestamp}.log"

    print(f"\n[调试信息] KVCache {offload_type} 卸载测试")
    print(f"  > Server日志文件: {server_log_file}")
    
    # 如果是磁盘模式，在临时目录中创建一个子目录用于存放缓存文件
    disk_cache_path = test_run_path / "disk_cache_storage"

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
    print(f"  > LMCache 配置文件 ({config_file}) 内容:\n{config_content}")

    # 设置环境变量
    original_env = os.environ.copy()
    env_vars_to_set = {
        "LMCACHE_USE_EXPERIMENTAL": "True",
        "LMCACHE_CONFIG_FILE": str(config_file),
        #"CUDA_VISIBLE_DEVICES": args.cuda_device
    }
    os.environ.update(env_vars_to_set)

    process = None
    log_file_handle = None
    
    try:
        # 启动 vLLM 服务
        print("\n--- 正在启动 vLLM KVCache 卸载服务 ---")
        log_file_handle = open(server_log_file, "w")
        
        vllm_cmd = [
            "vllm", "serve", args.model,
            "--port", str(args.port),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--max-model-len", str(args.max_model_len),
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
        ]
        
        print(f"  > vLLM 启动命令: {' '.join(vllm_cmd)}")
        
        # 启动 vLLM 服务进程，将 stdout 和 stderr 都重定向到日志文件
        process = subprocess.Popen(
            vllm_cmd, 
            env=os.environ.copy(), 
            stdout=log_file_handle, 
            stderr=subprocess.STDOUT
        )

        # 等待 vLLM 服务加载模型并开放端口
        print("  > 等待 vLLM 服务加载模型 (可能需要几分钟)...")
        _wait_for_port(args.host, args.port, timeout=1800)
        print("  > vLLM 服务已就绪。")

        # 将控制权和信息交给 "with" 语句块
        yield {
            "port": args.port,
            "host": args.host,
            "server_log_file": server_log_file,
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
                print(f"  > 进程 {process.pid} 被强制终止。")
        
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
    print(f"  > 正在等待端口 {host}:{port} ...")
    while True:
        try:
            with socket.create_connection((host, port), timeout=10):
                print(f"  > 端口 {port} 已开放。")
                return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() - start_time >= timeout:
                raise TimeoutError(f"等待端口 {port} 超时 ({timeout}s)。服务可能启动失败。")
            time.sleep(1)

def _get_long_prompt(args):
    """使用 tokenizer 创建一个指定 token 数量的超长提示"""
    print(f"  > 正在从 {args.model} 加载 tokenizer 以生成长提示...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"  > 警告：无法加载本地 tokenizer。将使用 'gpt2' 替代。错误: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (args.test_prompt_tokens - 10) # 留点余量
    long_context = tokenizer.decode(long_context_tokens)
    question = "\n\nSummarize the text above in exactly one sentence."
    prompt = long_context + question
    
    final_tokens = tokenizer.encode(prompt)
    if len(final_tokens) > args.max_model_len:
        print(f"  > 警告：生成的提示 ({len(final_tokens)} tokens) 超过了最大长度 ({args.max_model_len})。")
        prompt = tokenizer.decode(final_tokens[:args.max_model_len-50])

    print(f"  > 已生成长提示，Token 数量: {len(tokenizer.encode(prompt))}")
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
                    print(f"  > 收到第一个 token: '{chunk.choices[0].delta.content}'", end="", flush=True)
                else:
                    print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n  > 流式响应结束。")
        if first_token_time is None:
            raise RuntimeError("流式响应中没有收到任何内容。")
        return first_token_time - start_time
    except Exception as e:
        print(f"\n  > 查询时发生错误: {e}")
        return 9999.0

def _run_ttft_test(service_info, args):
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
    print(f"  > 服务地址: {base_url}")

    # 准备 OpenAI 客户端
    client = OpenAI(api_key="dummy-key", base_url=base_url)
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"  > 成功连接到 vLLM 服务, 模型 ID: {model_id}")
    except Exception as e:
        print(f"❌ 无法连接到 vLLM 服务: {e}。请检查服务日志: {service_info['server_log_file']}")
        return False

    # 准备长提示语
    prompt, _ = _get_long_prompt(args)

    # --- 冷缓存查询 ---
    print("\n--- [测试] 正在进行第一次查询 (冷缓存)... ---")
    cold_ttft = query_and_measure_ttft(client, model_id, prompt)
    print(f"\n✅ 冷缓存 TTFT: {cold_ttft:.3f} 秒")
    
    if cold_ttft <= 0:
        print("❌ 冷缓存查询太快了，可能 KVCache 未生效或 prompt 太短。")
        return False

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
    if warm_ttft >= (cold_ttft / 10):
        print(f"❌ 热缓存 ({warm_ttft:.3f}s) 不够快，没有达到冷缓存 ({cold_ttft:.3f}s) 的 1/10。")
        return False
    
    print(f"✅ KVCache {test_type} 卸载测试通过!")
    return True


# --- 两个独立的测试用例 ---

def test_kvcache_cpu_offload(test_run_path, args):
    """
    测试用例 1: 验证 CPU RAM 卸载。
    """
    print("\n" + "#"*70)
    print("### 开始测试用例 1: CPU KVCache 卸载 ###")
    print("#"*70)
    
    try:
        # "with" 语句会启动服务，并在代码块结束时自动关闭服务
        with vllm_offload_service("cpu", test_run_path, args) as service_info:
            result = _run_ttft_test(service_info, args)
            print("\n### 测试用例 1: CPU KVCache 卸载 完成 ###")
            return result
    except Exception as e:
        print(f"❌ CPU KVCache 卸载测试失败: {e}")
        return False


def test_kvcache_disk_offload(test_run_path, args):
    """
    测试用例 2: 验证 磁盘 (Disk) 卸载。
    """
    print("\n" + "#"*70)
    print("### 开始测试用例 2: 磁盘 KVCache 卸载 ###")
    print("#"*70)
    
    try:
        # "with" 语句会启动一个 新的服务，并在代码块结束时自动关闭服务
        with vllm_offload_service("disk", test_run_path, args) as service_info:
            result = _run_ttft_test(service_info, args)
            print("\n### 测试用例 2: 磁盘 KVCache 卸载 完成 ###")
            return result
    except Exception as e:
        print(f"❌ 磁盘 KVCache 卸载测试失败: {e}")
        return False


def create_test_directory(base_temp_dir, test_name):
    test_dir = Path(base_temp_dir) / f"kv_offload_{test_name}_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def main():
    args = parse_args()
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("vLLM KVCache 卸载测试开始")
    print("=" * 80)
    print(f"模型路径: {args.model}")
    print(f"host: {args.host}, port: {args.port}")
    print(f"最大模型长度: {args.max_model_len}")
    print(f"测试提示 tokens: {args.test_prompt_tokens}")
    print(f"测试 CPU offload: {args.test_cpu}")
    print(f"测试 Disk offload: {args.test_disk}")
    print(f"Server日志目录: {log_dir}")
    print("=" * 80)
    
    test_results = []
    tests_to_run = []
    
    if args.test_cpu:
        tests_to_run.append(("CPU KVCache offload", test_kvcache_cpu_offload))
    if args.test_disk:
        tests_to_run.append(("Disk KVCache offload", test_kvcache_disk_offload))
    
    if not tests_to_run:
        print("❌ 没有选择要运行的测试，请至少选择 --test-cpu 或 --test-disk")
        return 1
    
    base_temp_dir = tempfile.mkdtemp(prefix="kv_offload_test_")
    print(f"临时目录: {base_temp_dir}")
    
    try:
        for test_name, test_func in tests_to_run:
            print(f"\n> 开始执行: {test_name}")
            try:
                test_dir = create_test_directory(base_temp_dir, test_name.lower().replace(" ", "_"))
                result = test_func(test_dir, args)
                test_results.append((test_name, result))
                if result:
                    print(f"✅ {test_name} - 通过")
                else:
                    print(f"❌ {test_name} - 失败")
            except Exception as e:
                print(f"❌ {test_name} - 异常: {e}")
                test_results.append((test_name, False))
        
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
        
        print(f"\n测试完成：通过率：{pass_rate:.1f}% ({passed}/{total})")
        
    finally:
        try:
            shutil.rmtree(base_temp_dir)
        except Exception as e:
            print(f"警告：清理临时目录失败: {e}")


if __name__ == "__main__":
    sys.exit(main())