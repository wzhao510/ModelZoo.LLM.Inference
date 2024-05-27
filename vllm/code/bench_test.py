import os
import sys
import subprocess
from utils import get_params,write_txt



def run_benchmark(model_name, num_prompt):
    model_config = get_params(model_name)

    model_path = model_config["model_path"]
    tensor_parallel_size = model_config["c-eval_param"]["tensor_parallel_size"]
    dtype = model_config["c-eval_param"]["dtype"]
    gpu_memory_utilization = model_config["c-eval_param"]["gpu_memory_utilization"]
    task_name = model_config["c-eval_param"]["task_name"]
    batch_size = model_config["c-eval_param"]["batch_size"]

    c_eval_cmd = f'python ./code/src/benchmark_throughput.py  --model={model_path} --enforce-eager \
                --backend=vllm --max-model-len 2048 --num-prompts {num_prompt} --trust-remote-code \
                --input-len 1024 --output-len 1024 --tensor-parallel-size {tensor_parallel_size}'
    
    print(c_eval_cmd)
    use_cmd = True
    if use_cmd:
        os.system(c_eval_cmd)
    else:
        ret = subprocess.run(c_eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        data = ret.stdout
        data_name = model_path.replace("/", "_")
        write_txt(f"{data_name}.txt", data)
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
    modelname = sys.argv[1]
    num_prompt = sys.argv[2] if len(sys.argv) > 2 else 24
    run_benchmark(modelname, num_prompt)