import contextlib
import json
import os
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

from common import (
    assert_cache_speedup,
    dump_file as _dump_file,
    popen_own_group,
    query_and_measure_ttft,
    terminate_process_group,
    wait_for_free_gpu_memory,
    wait_for_port as _wait_for_port,
)

# ==============================================================================
# 全局实际配置参数
# ==============================================================================
VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8300
MOONCAKE_MASTER_PORT = 50251
MOONCAKE_METRICS_PORT = 19003
MAX_MODEL_LEN = 16384
TEST_PROMPT_TOKENS = 15000
CUDA_DEVICE_ID = "0"
# ==============================================================================


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
        # --metrics_port: mooncake_master 默认会额外起一个 HTTP 指标服务器监听
        # 9003(--port 只控制 RPC 服务端口,管不到它)。这台机器上跑了几十个共享
        # 容器且普遍 --net=host,9003 被别的租户的进程占着,导致 "Failed to start
        # master admin server on port 9003" 直接让整个 master 进程退出(跟本仓库
        # 之前排查 LMCache controller 撞 9000 端口是同一类问题;lsof/ss 因为看不到
        # 别的容器里的进程而显示"未占用",但直接对 9003 做 bind() 测试能实锤端口
        # 确实被占着)。曾经尝试过 --enable_metric_reporting=false 想干脆关掉这个
        # HTTP 服务器,但实测这个开关只影响是否上报数据,不影响端口 bind 本身
        # (日志里 enable_metric_reporting=0 之后照样打印 "HTTP metrics server
        # started on port 9003" 然后 bind 失败),所以还是得老老实实挪端口。
        #
        # --port(RPC 服务端口)本身也踩过同一类坑,而且更隐蔽:之前 MOONCAKE_MASTER_
        # PORT 写死是 50051,同样在这台 --net=host 共享主机上被别的租户占用。跟上面
        # 9003 的区别是,master 对 RPC 端口 bind 失败时完全不打印任何 ERROR 日志——
        # master.cpp 里 server.start() 是在独立线程里跑的,外层只是原样把返回码交
        # 还给 main() 就退出,一行报错都没有,日志看起来就是正常打印完启动参数之后
        # 紧跟着 "Task cleanup thread stopped",跟正常关闭没法用日志区分。master 会
        # 在起来后 ~100ms 内就悄悄退出,而 vLLM EngineCore 要等模型加载完(通常
        # 60~90+ 秒后)才第一次真正尝试连它,这时候面对的是一个早就已经不存在的
        # 进程,报出来的是 "RPC call failed: End of file"(创建 client 阶段,内部
        # 重试 20 次)或 "RPC call failed: invalid rpc arg"(MountSegment 阶段,不
        # 重试)。这两个报错乍一看很像 yalantinglibs/coro_rpc 序列化层的偶发性
        # bug,之前也确实在这个方向上排查了很久(给 coro_rpc 加源码级调试日志、
        # 抓包比对字节流等),最后是靠一个跟 popen_own_group 完全一致的最小复现
        # 脚本坐实的:换成非默认端口能稳定运行数分钟,换回 50051 就 100% 复现秒退,
        # `ss -tlnp` 也确认了 50051 当时确实已经被监听。只改端口(50051→50251)、
        # 其它代码不动,原本必现的失败就变成了必然通过。换成 50251 就是照搬这个
        # 结论,选一个当前空闲、不容易再撞车的端口。
        master_cmd = ["mooncake_master", "--port", str(MOONCAKE_MASTER_PORT),
                      "--metrics_port", str(MOONCAKE_METRICS_PORT)]
        if mode == "disk":
            master_cmd.append("--enable_offload=true")

        print(f"\n--- Starting mooncake_master: {' '.join(master_cmd)} ---")
        master_proc = popen_own_group(master_cmd, stdout=open(master_log, "w"), stderr=subprocess.STDOUT)
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

            client_proc = popen_own_group(client_cmd, env=client_env, stdout=open(client_log, "w"), stderr=subprocess.STDOUT)
            # 给予 client 几秒钟的初始化与 master 握手的时间
            time.sleep(3)

        yield

    finally:
        print("\n--- Tearing down Mooncake infrastructure ---")
        # Stop the client before the master (avoids the client spending its
        # shutdown window retrying a connection to an already-dead master).
        terminate_process_group(client_proc, "mooncake_client")
        terminate_process_group(master_proc, "mooncake_master")


@contextlib.contextmanager
def vllm_mooncake_service(test_mode, tmp_path_factory):
    """
    配置并启动使用 MooncakeStoreConnector 的 vLLM 服务。
    """
    test_run_path = tmp_path_factory.mktemp(f"mooncake_{test_mode}_test")
    config_file = test_run_path / "mooncake_config.json"
    log_file = test_run_path / "vllm_server.log"

    # 生成实际的 JSON 配置
    # protocol: 用 RDMA 起 MooncakeStoreConnector 的 client 时会反复
    # "Failed to create client on port ..., retry N/20" 最终 "RPC call failed:
    # End of file"。曾经怀疑是这台机器 GPUDirect RDMA 有问题(类比另一台 MetaX
    # 机器 10.13.106.39 上 mooncake PD 测试查到的显存注册失败),但对照了
    # test_mooncake_1p2d_tp2.py 用的 MooncakeConnector 源码
    # (vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)
    # 发现它默认协议也是 "rdma"、而且在这台机器上跑得完全正常——PD 和 offload
    # 底层用的是同一个 TransferEngine,PD 能用 RDMA 说明这台机器的 RDMA 传输本身
    # 没问题。区别在于 MooncakeStoreConnector 比 PD 多一层"向 mooncake_master
    # 注册成 store 客户端"的 RPC 握手(PD 是纯点对点,不需要 master),问题出在这层
    # 注册握手上,不是 RDMA 传输层。曾经尝试切到 "tcp" 想绕开,但这条路更差 ——
    # TCP 模式在 import 阶段就直接因为缺 libcudart.so.12 崩溃(这个容器只有 CUDA
    # 11.x 的库,v12 完全没有),所以改回 "rdma"(与 PD 保持一致,也是唯一验证过
    # 能跑通的协议),转而排查 Store 客户端注册握手本身为什么失败。
    mooncake_config = {
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": f"127.0.0.1:{MOONCAKE_MASTER_PORT}",
        "protocol": "rdma",
        "device_name": "mlx5_0",
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
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICE_ID,
        "MOONCAKE_CONFIG_PATH": str(config_file),
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
            
            process = popen_own_group(vllm_cmd, env=env, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
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
        terminate_process_group(process, "vLLM (mooncake)")
        if process is not None:
            wait_for_free_gpu_memory(CUDA_DEVICE_ID)


def _get_long_prompt():
    print(f"Loading tokenizer from {VLLM_MODEL_PATH} to build a long prompt...")
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH, trust_remote_code=True)
    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (TEST_PROMPT_TOKENS - 10)
    prompt = tokenizer.decode(long_context_tokens) + "\n\nSummarize exactly one sentence."
    return prompt


def _run_ttft_test(service_info):
    base_url = f"http://{service_info['host']}:{service_info['port']}/v1"
    client = OpenAI(api_key="dummy-key", base_url=base_url)
    model_id = client.models.list().data[0].id
    prompt = _get_long_prompt()

    print("\n--- Cold Cache Query (Storing to Mooncake) ---")
    cold_ttft, cold_output = query_and_measure_ttft(client, model_id, prompt, max_tokens=20)
    print(f"Cold TTFT: {cold_ttft:.3f}s")

    time.sleep(3) # 给 Mooncake 异步传输引擎一点时间将 Cache 刷入共享池或磁盘

    print("\n--- Warm Cache Query (Loading from Mooncake) ---")
    warm_ttft, warm_output = query_and_measure_ttft(client, model_id, prompt, max_tokens=20)
    print(f"Warm TTFT: {warm_ttft:.3f}s")

    # 用关键字包含（cold 输出前缀是否出现在 warm 输出中）代替逐字完全一致：
    # 同一进程内 temperature=0.0 通常稳定，但要求逐字节完全相同比 KV cache
    # 本身的一致性承诺更严格，容易因末尾细微差异产生误报。
    #
    # max_warm_ratio 从 0.7 放宽到 0.9：不同 GPU 型号/驱动上，命中缓存省下的计算
    # 时间占总 TTFT 的比例会不一样(见 test_kv_share_lmcache.py 里的同类说明)，防止
    # "没有真正命中"的主要防线是上面的关键字包含校验，这里只是在此基础上再确认
    # "确实变快了"。
    assert_cache_speedup("Mooncake KV offload", cold_ttft, warm_ttft, cold_output, warm_output,
                          max_warm_ratio=0.9)


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