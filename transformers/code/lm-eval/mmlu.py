import os
import sys
import subprocess
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils import get_params,get_model_performeance_filename,update_json_file


def run_eval(model_name):
    model_config = get_params(model_name)
    performance_json = get_model_performeance_filename(model_name)
    model_path = model_config["model_path"]
    tensor_parallel_size = model_config["param"]["tensor_parallel_size"]
    dtype = model_config["param"]["dtype"]
    task_name = "mmlu"
    batch_size = model_config["param"]["batch_size"]
    if tensor_parallel_size > 1:
        parallelize = True
    else:
        parallelize = False

    eval_cmd = f'lm_eval --model hf \
    --model_args pretrained={model_path},parallelize={parallelize},dtype={dtype}\
    --trust_remote_code \
    --tasks {task_name} \
    --batch_size {batch_size}'
    
    print(eval_cmd)
    use_cmd = True
    if use_cmd:
        os.system(eval_cmd)
    else:
        ret = subprocess.run(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        data = ret.stdout
        # data_name = model_path.replace("/", "_")
        # write_txt(f"./{date_str}/{data_name}.txt", data)
        data_list = data.split("\n")
        acc = 0
        std = 0
        #if len(data_list) > 3:
        acc_val_str = data_list[3]
        acc_list = acc_val_str.split("|")
        if len(acc_list) >=4:
            acc = acc_list[-4]
            std = acc_list[-2]
        print(f"result: {acc}±{std}")
        update_json_file(performance_json,task_name,dtype,float(acc))


if __name__ == '__main__':
    modelname = sys.argv[1]
    run_eval(modelname)
