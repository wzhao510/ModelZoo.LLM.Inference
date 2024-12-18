import os
import argparse
import platform
import sys
import subprocess
from utils import get_params,write_txt, get_vllm_version

def is_int(value):
    try:
        int_value = int(value)
        return True
    except ValueError:
        return False


def run_benchmark(model_name, num_prompt, input_len, output_len, is_batched, enforce_eager, num_scheduler_steps):
    model_config = get_params(model_name)
    vllm_version = get_vllm_version()
    print(f"vLLM version: {vllm_version}")
    if vllm_version is None:
        raise ValueError("Cannot get vllm_version.")

    model_path = model_config["model_path"]
    lora_path = model_config.get("lora_path", None)
    tensor_parallel_size = model_config["c-eval_param"]["tensor_parallel_size"]
    pipeline_parallel_size = model_config["c-eval_param"].get("pipeline_parallel_size")
    async_engine = model_config.get("async_engine")
    dtype = model_config["c-eval_param"]["dtype"]
    gpu_memory_utilization = model_config["c-eval_param"]["gpu_memory_utilization"]
    task_name = model_config["c-eval_param"]["task_name"]
    batch_size = model_config["c-eval_param"]["batch_size"]

    enable_profile = os.getenv("MX_VLLM_ENABLE_PROFILE", None)
    enable_profile= False if enable_profile is None else True


    if is_int(is_batched) and int(is_batched) == 0:
        if vllm_version.startswith("0.4.0"):
            c_eval_cmd = f'python ./code/src/benchmark_throughput.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size} '
        elif vllm_version.startswith("0.5"):
             c_eval_cmd = f'python ./code/src/benchmark_throughput_0.5.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
        elif vllm_version.startswith("0.6"):
            extra_args = ""
            if pipeline_parallel_size is not None:
                extra_args = f"{extra_args} --pipeline-parallel-size={pipeline_parallel_size}"
            if async_engine is not None and async_engine:
                extra_args = f"{extra_args} --async-engine"
            print(f"extra_args: {extra_args}")
            c_eval_cmd = f'python ./code/src/benchmark_throughput_0.6.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size} \
                        {extra_args}'
    else:
        if vllm_version.startswith("0.4.0"):
            c_eval_cmd = f'python ./code/src/benchmark_throughput_batched.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
        elif vllm_version.startswith("0.5"):
            c_eval_cmd = f'python ./code/src/benchmark_throughput_batched_0.5.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
        else:
            c_eval_cmd = f'python ./code/src/benchmark_throughput_batched_0.6.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
    
    if lora_path is not None and os.path.isdir(lora_path):
        c_eval_cmd += f" --lora_path {lora_path}"
    
    if enable_profile:
        c_eval_cmd += " --enable-profile"
    
    # if arm, we set enforce_eager=false to turn on CUDA_GRAPH for better performance.
    # if you want to test another option, please modify this code.
    machine = platform.machine()
    if enforce_eager:
        c_eval_cmd += f" --enforce-eager"

    if num_scheduler_steps is not None and vllm_version.startswith("0.6"):
        c_eval_cmd += f" --num-scheduler-steps={num_scheduler_steps}"

    print(c_eval_cmd)
    
    os.system(c_eval_cmd)

if __name__ == '__main__':
    # modelname = sys.argv[1]
    # num_prompt = sys.argv[2] if len(sys.argv) > 2 else 24
    # input_len =  sys.argv[3] if len(sys.argv) > 3 else 1024
    # output_len =  sys.argv[4] if len(sys.argv) > 4 else 1024
    # is_batched = sys.argv[5] if len(sys.argv) > 5 else 0
    # enforce_eager = sys.argv[6] if len(sys.argv) > 6 else None
    # num_scheduler_steps = int(sys.argv[7]) if len(sys.argv) > 7 else None
    # run_benchmark(modelname, num_prompt, input_len, output_len, is_batched, enforce_eager, num_scheduler_steps)

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
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    args = parser.parse_args()
    run_benchmark(args.model, args.num_prompts, args.input_len, args.output_len, args.batched_test, args.enforce_eager, args.num_scheduler_steps)
