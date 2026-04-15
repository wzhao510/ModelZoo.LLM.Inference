import os
import argparse
import subprocess

from utils import get_params,write_txt, get_vllm_version, get_cpu_info, get_cpu_vendor_id, get_common_params,get_json_string
from src.batched_test.gpu_manager import GPUManager
from typing import Any


from vllm.config import (
    ModelConfig,
    CacheConfig,
    LoRAConfig,
    ParallelConfig,
    SchedulerConfig,
    StructuredOutputsConfig,
    SpeculativeConfig,
    MultiModalConfig,
    ObservabilityConfig
)
from pydantic.dataclasses import dataclass
from packaging import version

gpu_manager = GPUManager()

cpu_vendor_id_name_mapping = {
    "GenuineIntel":"Intel",
    "AuthenticAMD": "AMD",
    "HygonGenuine":"Hygon",
    "HygonAuthentic":"Hygon",
}

@dataclass
class FrontendConfig:
    lora_modules: str | None = None
    chat_template: str | None = None
    chat_template_content_format: str = "auto"
    port: int = 8000
    host: str = "127.0.0.1"

@dataclass
class OffloadConfig:
    pass

@dataclass
class BenchConfig:
    pass

class InferSched:
    pass 
        


def get_env_var(args, config):
    if args.gpu_device:
        gpu_name = args.gpu_device
    else:
        gpu_name = "Default"
        if gpu_manager.get_gpu_count() > 0:
            gpu_name = gpu_manager.get_gpu_name(0)
    if args.cpu_device:
        cpu_vendor_name = args.cpu_device
    else:
        cpu_vendor_id = get_cpu_vendor_id()
        cpu_vendor_name = cpu_vendor_id_name_mapping[cpu_vendor_id]
    print(f'{gpu_name=} {cpu_vendor_name=}')
    env_config = config["env_config"].get(gpu_name)
    common_env_config = get_common_params("env_config.json").get(gpu_name)
    assert env_config and common_env_config, f"Not support gpu {gpu_name}"
    # env_config优先级 高于comon_env_config
    print(f"common_env_config: {common_env_config.get(cpu_vendor_name)}")
    print(f"env_config: {env_config.get(cpu_vendor_name)}")
    gpu_env_config = common_env_config.get(cpu_vendor_name) | env_config.get(cpu_vendor_name)
    fixed_gpu_env_config = {}
    if gpu_env_config is not None:
        for key, value in gpu_env_config.items():
            if value != "disable":
                fixed_gpu_env_config[key] = value

    gpu_env_config = fixed_gpu_env_config
    if cpu_vendor_name == "Hygon" and gpu_name == "MXC588":
        gpu_count = gpu_manager.get_gpu_count()
        if gpu_count == 8:
            ids = [0,1,2,3,6,5,4,7]
        elif gpu_count == 16:
            ids = [0,1,2,3,6,5,4,7,10,11,8,9,14,13,12,15]
        else:
            ids = list(range(gpu_count))
        visible_devices = ','.join([str(id) for id in ids])
        gpu_env_config["CUDA_VISIBLE_DEVICES"] = visible_devices
        gpu_env_config["MACA_VISIBLE_DEVICES"] = visible_devices
    print(f'{gpu_env_config=}')
    return gpu_env_config, gpu_name, cpu_vendor_name


def revision_vllm_args(args: list, config) -> list:
    if config.get("revision_config") is None:
        return args
    
    revision_config = config.get("revision_config")
    revisioned_args = []
    for arg in args:
        revisioned_args.append(arg)
        arg = arg.replace("-", "_")
        vllm_version_requires = revision_config.get("vllm_version_requires")
        if vllm_version_requires:
            vllm_version = get_vllm_version()
            if vllm_version_requires.get(arg):
                value = vllm_version_requires.get(arg)
                if value.startswith(">="):
                    required_version = version.parse(value[2:])
                    if vllm_version < required_version:
                        revisioned_args.pop()
                elif value.startswith(">"):
                    required_version = version.parse(value[1:])
                    if vllm_version <= required_version:
                        revisioned_args.pop()
                elif value.startswith("<="):
                    required_version = version.parse(value[2:])
                    if vllm_version > required_version:
                        revisioned_args.pop()
                elif value.startswith("<"):
                    required_version = version.parse(value[1:])
                    if vllm_version >= required_version:
                        revisioned_args.pop()
                elif value == "==":
                    required_version = version.parse(value[2:])
                    if vllm_version != required_version:
                        revisioned_args.pop()
                else:
                    print(f"[WARN] invalid vllm version required config")
    return revisioned_args




def vllm_serve(args, config):
    # cmd parts
    cmd_parts = ["vllm serve"]
    # envs
    env_vars, gpu_name, cpu_vendor_name = get_env_var(args, config=config)
    env_vars_str = " ".join([f"{key}={value}" for key,value in env_vars.items()])
    print(f"{env_vars_str=}")
    # vllm args 
    vllm_args = []
    # trust-remote-code
    trust_remote_code = config["model_config"].get("trust_remote_code")
    if trust_remote_code is not None:
        if trust_remote_code:
            vllm_args.append("trust-remote-code")
        else:
            vllm_args.append("no-trust-remote-code")
    # async-scheduling
    async_scheduling = config["scheduler_config"].get("async_scheduling")
    if async_scheduling is not None:
        if async_scheduling:
            vllm_args.append("async-scheduling")
        else:
            vllm_args.append("no-async-scheduling")
    # enable-prefix-caching
    enable_prefix_caching = config["cache_config"].get("enable_prefix_caching")
    if enable_prefix_caching is not None:
        if enable_prefix_caching:
            vllm_args.append("enable-prefix-caching")
        else:
            vllm_args.append("no-enable-prefix-caching")
    # enable-auto-tool-choice
    enable_auto_tool_choice = config["frontend_config"].get("enable_auto_tool_choice")
    if enable_auto_tool_choice is not None:
        if enable_auto_tool_choice:
            vllm_args.append("enable-auto-tool-choice")
        else:
            vllm_args.append("no-enable-auto-tool-choice")
    # skip-mm-profiling
    skip_mm_profiling = config["multi_modal_config"].get("skip_mm_profiling")
    if skip_mm_profiling is not None:
        if skip_mm_profiling:
            vllm_args.append("skip-mm-profiling")
        else:
            vllm_args.append("no-skip-mm-profiling")

    cmd_parts.append(config["model_config"].get("model"))
    # vllm kwargs
    vllm_kwargs = {
        #"model": config["model_config"].get("model"),
        "tensor-parallel-size": config["parallel_config"].get("tensor_parallel_size"),
        "data-parallel-size": config["parallel_config"].get("data_parallel_size"),
        "max-model-len": config["model_config"].get("max_model_len"),
        "max-num-seqs": config["scheduler_config"].get("max_num_seqs"),
        "gpu-memory-utilization": config["cache_config"].get("gpu_memory_utilization"),
        "port": config["frontend_config"].get("port"),
        "tool-call-parser": config["frontend_config"].get("tool_call_parser"),
        "swap-space":config["cache_config"].get("swap_space"),
        "seed": config["model_config"].get("seed"),
        "max-num-batched-tokens": config["scheduler_config"].get("max_num_batched_tokens"),
        "reasoning-parser": config["structured_outputs_config"].get("reasoning_parser"),
        "mm-processor-cache-type": config["multi_modal_config"].get("mm_processor_cache_type"),
        "mm-encoder-tp-mode": config["multi_modal_config"].get("mm_encoder_tp_mode"),
        "mm-processor-cache-gb": config["multi_modal_config"].get("mm_processor_cache_gb"),
        "distributed-executor-backend": config["parallel_config"].get("distributed_executor_backend")
    }

    for value in vllm_args:
        cmd_parts.append(f"--{value}")

    for key, value in vllm_kwargs.items():
        if value is not None:
            cmd_parts.append(f"--{key} {value}")

    vllm_serve_cmd = " \\\n".join(cmd_parts)
    if config.get("speculative_config"):
        speculative_config_str = get_json_string(config.get("speculative_config"))
        vllm_serve_cmd = f"{vllm_serve_cmd} \\\n--speculative-config '{speculative_config_str}'"
    if config.get("multi_modal_config") and config["multi_modal_config"].get("limit_mm_per_prompt"):
        limit_mm_per_prompt_str = get_json_string(config["multi_modal_config"].get("limit_mm_per_prompt"))
        vllm_serve_cmd = f"{vllm_serve_cmd} \\\n--limit-mm-per-prompt '{limit_mm_per_prompt_str}'"

    if env_vars_str:
        vllm_serve_cmd = f"{env_vars_str} {vllm_serve_cmd}"
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    print("*" * width)
    print(vllm_serve_cmd)
    print("*" * width)
    filename = f"{gpu_name}_{cpu_vendor_name}_vllm_serve_cmd.txt"
    file_path = f"{args.model}/runtime/command/{filename}"
    print(f"{file_path=}")
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_txt(file_path, vllm_serve_cmd)
    if args.execute_command:
        subprocess.run(vllm_serve_cmd, shell=True)


def vllm_bench_serve(args, config):
    env_vars, gpu_name, cpu_vendor_name = get_env_var(args, config=config)
    # cmd parts
    cmd_parts = ["vllm bench serve"]
    bench_config = config["bench_config"]
    model_config = config["model_config"]
    frontend_config = config["frontend_config"]
    # vllm args
    vllm_args = []
    trust_remote_code = bench_config.get("trust_remote_code") or model_config.get("trust_remote_code")
    if trust_remote_code is not None:
        if trust_remote_code:
            vllm_args.append("trust-remote-code")
        else:
            vllm_args.append("no-trust-remote-code")

    ignore_eos = bench_config.get("ignore_eos")
    if ignore_eos is not None:
        if ignore_eos:
            vllm_args.append("ignore-eos")
            
    
    # vllm kwargs
    vllm_kwargs = {
        "model": model_config.get("model"),
        "port": bench_config.get("port") or frontend_config.get("port"),
        "host": bench_config.get("host") or frontend_config.get("host"),
        "backend": bench_config.get("backend", "vllm"),
        "endpoint": bench_config.get("endpoint"),
        "dataset-name": bench_config.get("dataset_name"),
        "num-prompts": bench_config.get("num_prompts"),
        "max-concurrency": bench_config.get("max_concurrency"),
        "random-prefix-len": bench_config.get("random-prefix-len"),
        "random-input-len": bench_config.get("random_input_len"),
        "random-output-len": bench_config.get("random_output_len"),
        "random-range-ratio": bench_config.get("random_range_ratio"),
        "random-mm-base-items-per-request": bench_config.get("random_mm_base_items_per_request"),
        "seed": bench_config.get("seed") or model_config.get("seed"),
    }

    for value in vllm_args:
        cmd_parts.append(f"--{value}")

    for key, value in vllm_kwargs.items():
        if value is not None:
            cmd_parts.append(f"--{key} {value}")
    vllm_bench_serve_cmd = " \\\n".join(cmd_parts)
    if bench_config.get("random_mm_limit_mm_per_prompt"):
        random_mm_limit_mm_per_prompt_str = get_json_string(bench_config.get("random_mm_limit_mm_per_prompt"))
        vllm_bench_serve_cmd = f"{vllm_bench_serve_cmd} \\\n--random-mm-limit-mm-per-prompt '{random_mm_limit_mm_per_prompt_str}'"
    if bench_config.get("random_mm_bucket_config"):
        random_mm_bucket_config_str = bench_config.get("random_mm_bucket_config")
        vllm_bench_serve_cmd = f"{vllm_bench_serve_cmd} \\\n--random-mm-bucket-config '{random_mm_bucket_config_str}'"
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    print("*" * width)
    print(vllm_bench_serve_cmd)
    print("*" * width)
    
    filename = f"{gpu_name}_{cpu_vendor_name}_vllm_bench_serve_cmd.txt"
    file_path = f"{args.model}/runtime/command/{filename}"
    print(f"{file_path=}")
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_txt(file_path, vllm_bench_serve_cmd)
    if args.execute_command:
        subprocess.run(vllm_bench_serve_cmd, shell=True)

    
def vllm_bench_throughput(args, config):
    env_vars, gpu_name, cpu_vendor_name = get_env_var(args, config=config)
    # cmd parts
    cmd_parts = ["vllm bench throughput"]
    bench_config = config["bench_config"]
    model_config = config["model_config"]
    frontend_config = config["frontend_config"]
    # vllm args
    vllm_args = []
    trust_remote_code = bench_config.get("trust_remote_code") or model_config.get("trust_remote_code")
    if trust_remote_code is not None and trust_remote_code:
        vllm_args.append("trust-remote-code")
    else:
        vllm_args.append("no-trust-remote-code")

    if bench_config.get("ignore-eos"):
        vllm_args.append(bench_config.get("ignore-eos"))
    
    # vllm kwargs
    vllm_kwargs = {
        "model": model_config.get("model"),
        "backend": bench_config.get("backend", "vllm"),
        "endpoint": bench_config.get("endpoint"),
        "input-len": bench_config.get("input_len"),
        "output-len": bench_config.get("output_len"),
        "dataset-name": bench_config.get("dataset_name"),
        "num-prompts": bench_config.get("num_prompts"),
        "random-prefix-len": bench_config.get("random-prefix-len"),
        "random-range-ratio": bench_config.get("random_range_ratio"),
        "seed": bench_config.get("seed") or model_config.get("seed")
    }

    for value in vllm_args:
        cmd_parts.append(f"--{value}")

    for key, value in vllm_kwargs.items():
        if value:
            cmd_parts.append(f"--{key} {value}")
    vllm_bench_throughput_cmd = " \\\n".join(cmd_parts)
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    print("*" * width)
    print(vllm_bench_throughput_cmd)
    print("*" * width)
    filename = f"{gpu_name}_{cpu_vendor_name}_vllm_bench_throughput_cmd.txt"
    file_path = f"{args.model}/runtime/command/{filename}"
    print(f"{file_path=}")
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_txt(file_path, vllm_bench_throughput_cmd)
    if args.execute_command:
        subprocess.run(vllm_bench_throughput_cmd, shell=True)

def main(args):
    config = get_params(args.model)
    if args.mode == "serve":
        vllm_serve(args, config=config)
    if args.mode == "bench_serve":
        vllm_bench_serve(args, config=config)
    if args.mode == "bench_throughput":
        vllm_bench_throughput(args, config=config)

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="models/Qwen/Qwen2.5-0.5B-Instruct",
                        help="model path to be tested")
    parser.add_argument("--mode",
                        type=str,
                        default="serve",
                        choices=["serve", "bench_serve", "bench_throughput"],
                        help="specify the perf mode, possible choices: [serve, bench_serve, bench_throughput]"
                        )
    parser.add_argument("--execute-command",
                        action="store_true",
                        help="whether to execute the generated command. Defaults to true"
                        )
    parser.add_argument("--gpu-device",
                        type=str,
                        help="select gpu device type, if no device type is specified,\
                            it will be automatically detected based on the runtime environment,\
                                possible choices: [MXC500, MXC550, MXC558, MXC600]")
    parser.add_argument("--cpu-device",
                        type=str,
                        choices=["Intel", "AMD", "Hygon"],
                        help="select cpu device type, if no device type is specified,\
                            it will be automatically detected based on the runtime environment,\
                                possible choices: [Intel, AMD, Hygon]")
    args = parser.parse_args()
    main(args)