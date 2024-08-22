import os
import sys
import subprocess
from utils import get_params,write_txt, get_vllm_version

def is_int(value):
    try:
        int_value = int(value)
        return True
    except ValueError:
        return False


def run_benchmark(model_name, num_prompt, input_len, output_len, is_batched):
    model_config = get_params(model_name)
    vllm_version = get_vllm_version()
    print(f"vLLM version: {vllm_version}")
    if vllm_version is None:
        raise ValueError("Cannot get vllm_version.")

    model_path = model_config["model_path"]
    lora_path = model_config.get("lora_path", None)
    tensor_parallel_size = model_config["c-eval_param"]["tensor_parallel_size"]
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
        else:
             c_eval_cmd = f'python ./code/src/benchmark_throughput_0.5.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
    else:
        if vllm_version.startswith("0.4.0"):
            c_eval_cmd = f'python ./code/src/benchmark_throughput_batched.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
        else:
            c_eval_cmd = f'python ./code/src/benchmark_throughput_batched_0.5.0.py  --model={model_path}  \
                        --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size}'
    
    if lora_path is not None and os.path.isdir(lora_path):
        c_eval_cmd += f" --lora_path {lora_path}"
    
    if enable_profile:
            c_eval_cmd += " --enable-profile"
    
    print(c_eval_cmd)
    use_cmd = True
    os.system(c_eval_cmd)

if __name__ == '__main__':
    modelname = sys.argv[1]
    num_prompt = sys.argv[2] if len(sys.argv) > 2 else 24
    input_len =  sys.argv[3] if len(sys.argv) > 3 else 1024
    output_len =  sys.argv[4] if len(sys.argv) > 4 else 1024
    is_batched = sys.argv[5] if len(sys.argv) > 5 else 0
    run_benchmark(modelname, num_prompt, input_len, output_len, is_batched)
