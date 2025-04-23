# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os
import argparse
import platform
from utils import get_params, get_vllm_version
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

def is_int(value):
    try:
        int_value = int(value)
        return True
    except ValueError:
        return False


def run_benchmark_mutlimoda(args, model_config):
    script_file = "./code/src/benchmark_mutlimoda.py"
    script_args = [
        "--model={}".format(model_config["model_path"]),
        "--model-type={}".format(model_config["model_type"]),
        "--max-model-len=8192",
        "--num-prompts={}".format(args.num_prompts),
        "--trust-remote-code",
        "--dtype={}".format(model_config["model_param"]["dtype"]),
        "--input-len={}".format(args.input_len),
        "--output-len={}".format(args.output_len),
        "--tensor-parallel-size={}".format(model_config["model_param"]["tensor_parallel_size"]),
        "--num-scheduler-steps={}".format(args.num_scheduler_steps),
    ]

    if args.enforce_eager:
        script_args.append("--enforce-eager")
    if args.batched_test:
        script_args.append("--benchmark-all")
    if args.distributed_executor_backend is not None:
        script_args.append(f"--distributed-executor-backend={args.distributed_executor_backend}")

    enable_profile = os.getenv("MX_VLLM_ENABLE_PROFILE", "").lower()
    if enable_profile in ("yes", "true", "t", "y", "1"):
        script_args.append("--enable-profile")

    cmd = "python3 {} {}".format(script_file, " ".join(script_args))
    print(cmd)
    if os.system(cmd) != 0:
        exit(1)


def run_benchmark(args):
    model_config = get_params(args.model)
    vllm_version = get_vllm_version()
    print(f"vLLM version: {vllm_version}")
    if vllm_version is None:
        raise ValueError("Cannot get vllm_version.")
    
    if model_config.get("mutlimoda", False):
        run_benchmark_mutlimoda(args, model_config)
        return

    model_path = model_config["model_path"]
    lora_path = model_config.get("lora_path", None)
    tensor_parallel_size = model_config["c-eval_param"]["tensor_parallel_size"]
    pipeline_parallel_size = model_config["c-eval_param"].get("pipeline_parallel_size")
    async_engine = model_config.get("async_engine")
    dtype = model_config["c-eval_param"]["dtype"]
    gpu_memory_utilization = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else model_config["c-eval_param"]["gpu_memory_utilization"]
    
    enable_profile = os.getenv("MX_VLLM_ENABLE_PROFILE", None)
    enable_profile= False if enable_profile is None else True
    if args.speculative_model is  None:
        benchmark_cmd = f'python ./code/src/benchmark_throughput.py  --model={model_path}  \
                        --backend=vllm --max-model-len {args.max_model_len} --num-prompts {args.num_prompts} --trust-remote-code --dtype {dtype} \
                        --input-len {args.input_len} --output-len {args.output_len} --tensor-parallel-size {tensor_parallel_size} --gpu-memory-utilization {gpu_memory_utilization}'
    else:
        benchmark_cmd = f'python ./code/src/benchmark_throughput.py  --model={model_path}  \
                        --backend=vllm --max-model-len {args.max_model_len} --num-prompts {args.num_prompts} --trust-remote-code --dtype {dtype} \
                        --input-len {args.input_len} --output-len {args.output_len} --tensor-parallel-size {tensor_parallel_size} --gpu-memory-utilization {gpu_memory_utilization} \
                        --speculative-model {args.speculative_model} \
                        --num-speculative-tokens {args.num_speculative_tokens} \
                        --ngram-prompt-lk-max {args.ngram_prompt_lk_max}'
    if args.batched_test:
        benchmark_cmd += " --batched-test"  

    if args.quantization is not None:
        benchmark_cmd += f" --quantization {args.quantization}"  

    if async_engine is not None and async_engine:
        benchmark_cmd += " --async-engine"
    
    if pipeline_parallel_size is not None:
        benchmark_cmd += f" --pipeline-parallel-size={pipeline_parallel_size}"

    if lora_path is not None and os.path.isdir(lora_path):
        benchmark_cmd += f" --lora_path {lora_path}"
    
    if enable_profile:
        benchmark_cmd += " --enable-profile"

    if args.distributed_executor_backend is not None:
        benchmark_cmd += f" --distributed-executor-backend {args.distributed_executor_backend}"
    
    # if arm, we set enforce_eager=false to turn on CUDA_GRAPH for better performance.
    # if you want to test another option, please modify this code.
    if args.enforce_eager:
        benchmark_cmd += f" --enforce-eager"
    
    if args.disable_sliding_window:
        benchmark_cmd += f" --disable-sliding-window"

    if args.num_scheduler_steps is not None:
        benchmark_cmd += f" --num-scheduler-steps={args.num_scheduler_steps}"

    print(benchmark_cmd)
    
    os.system(benchmark_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=32,
                        help="Number of prompts for throughput test")
    parser.add_argument("--input-len",
                        type=int,
                        default=1024,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=1024,
                        help="Output length for each request.")
    parser.add_argument("--batched-test",
                        action="store_true",
                        help="Test 35 case but load model once.")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=2048,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument(
        "--disable-sliding-window",
        action='store_true',
        default=False,
        help="Disable sliding window for vLLM backend.")
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument("--enable-chunked-prefill",
                        action="store_true",
                        help="enforce chunked prefill")
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument("--speculative-model",
                        type=str,
                        default=None,
                        help="speculative model path")
    
    parser.add_argument("--num-speculative-tokens",
                        type=int,
                        default=4,
                        help="")
    parser.add_argument("--ngram-prompt-lk-max",
                        type=int,
                        default=4,
                        help="ngram_prompt_lookup_max")
    args = parser.parse_args()
    run_benchmark(args)
