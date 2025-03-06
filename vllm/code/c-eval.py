# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os
import sys
import subprocess
import argparse
from utils import get_params,write_txt, get_vllm_version



def run_eval(model_name, enforce_eager=True):
    model_config = get_params(model_name)
    vllm_version = get_vllm_version()
    print(f"vLLM version: {vllm_version}")
    if vllm_version is None:
        raise ValueError("Cannot get vllm_version.")
    

    model_path = model_config["model_path"]
    tensor_parallel_size = model_config["c-eval_param"]["tensor_parallel_size"]
    dtype = model_config["c-eval_param"]["dtype"]
    gpu_memory_utilization = model_config["c-eval_param"]["gpu_memory_utilization"]
    task_name = model_config["c-eval_param"]["task_name"]
    batch_size = model_config["c-eval_param"]["batch_size"]

    if vllm_version.startswith("0.4.0"):
        c_eval_cmd = f'lm_eval --model vllm \
            --model_args pretrained={model_path},tensor_parallel_size={tensor_parallel_size},dtype={dtype},enforce_eager={enforce_eager},max_model_len=2048,trust_remote_code=True,gpu_memory_utilization={gpu_memory_utilization} \
            --trust_remote_code \
            --tasks {task_name} \
            --batch_size {batch_size} \
            --output_path results \
            --log_samples '
    else:
         c_eval_cmd = f'lm_eval --model vllm \
            --model_args pretrained={model_path},tensor_parallel_size={tensor_parallel_size},dtype={dtype},enforce_eager={enforce_eager},max_model_len=2048,trust_remote_code=True,gpu_memory_utilization={gpu_memory_utilization},distributed_executor_backend=ray \
            --trust_remote_code \
            --tasks {task_name} \
            --batch_size {batch_size}'
    
    print(c_eval_cmd)
    use_cmd = True
    if use_cmd:
        if os.system(c_eval_cmd) != 0:
            exit(1)
    else:
        ret = subprocess.run(c_eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        data = ret.stdout
        data_name = model_path.replace("/", "_")
        write_txt(f"./20240423/{data_name}.txt", data)
        data_list = data.split("\n")
        acc = 0
        std = 0
        if len(data_list) > 3:
            acc_val_str = data_list[-3]
            
            acc_list = acc_val_str.split("|")
            if len(acc_list) >=4:
                acc = acc_list[-4]
                std = acc_list[-2]
        print(f"result: {acc}±{std}")
    # import pdb;pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CEval test")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    args = parser.parse_args()
    run_eval(args.model, args.enforce_eager)
