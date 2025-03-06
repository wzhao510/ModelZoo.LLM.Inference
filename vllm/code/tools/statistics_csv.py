# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os
import sys
import pandas as pd


def convert_to_ms(time_str):
    if time_str.endswith('ms'):
        return float(time_str[:-2])
    elif time_str.endswith('us'):
        return float(time_str[:-2]) / 1000
    elif time_str.endswith('s'):
        return float(time_str[:-1]) * 1000
    
    else:
        raise ValueError(f"Unknown time format: {time_str}")


def get_json(file_path):
    # file_path = 'profile_data.csv'
    df = pd.read_csv(file_path)
    df['CUDA_total(ms)'] = df['CUDA total'].apply(convert_to_ms)

    summary = {
        'gemm': {"time":0, "call":0 ,"percent":0},
        'mccl': {"time":0, "call":0 ,"percent":0},
        'flash-attn': {"time":0, "call":0 ,"percent":0},
        'vllm:paged_attn_v1': {"time":0, "call":0 ,"percent":0},
        'vllm:paged_attn_v2': {"time":0, "call":0 ,"percent":0},
        'vllm:rotary_embeded': {"time":0, "call":0 ,"percent":0},
        'vllm:reshape_and_cache': {"time":0, "call":0 ,"percent":0},
        "vllm:layernorm" : {"time":0, "call":0 ,"percent":0},
        'vllm:act_and_mul': {"time":0, "call":0 ,"percent":0},
        "other": {"time":0, "call":0 ,"percent":0},
    }

    name_mapping = {
        # c500 mapping
        "mcblas__Mck" : "gemm",
        "mcclKernel" : "mccl",
        "void flash_fwd" : "flash-attn",
        "void vllm::paged_attention_v1" : "vllm:paged_attn_v1",
        "void vllm::paged_attention_v2" : "vllm:paged_attn_v2",
        "void vllm::rotary_embedding_kernel" : "vllm:rotary_embeded",
        "void vllm::reshape_and_cache_kernel" : "vllm:reshape_and_cache",
        "void vllm::fused_add_rms_norm_kernel" : "vllm:layernorm",
        "void vllm::act_and_mul_kernel" : "vllm:act_and_mul" 
    }


    for index, row in df.iterrows():
        name = row['Name']
        time_in_ms = row['CUDA_total(ms)']
        time_of_call = row["# of Calls"]
        time_in_percent = row["Self CUDA %"]
        flag = False
        for prefix_name, kernel_name in name_mapping.items():
            if name.startswith(prefix_name):
                summary[kernel_name]["time"] += time_in_ms
                summary[kernel_name]["call"] += int(time_of_call)
                summary[kernel_name]["percent"] += float(time_in_percent.replace("%",""))
                flag = True
        
        if not flag:
            kernel_name = "other"
            summary[kernel_name]["time"] += time_in_ms
            summary[kernel_name]["call"] += int(time_of_call)
            summary[kernel_name]["percent"] += float(time_in_percent.replace("%",""))

    for key, value in summary.items():
        value['time'] = round(value['time'], 3)
        value['percent'] = round(value['percent'], 3)


    return summary

def main(dataset_path):
    all_data_dict = {}


    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        print(csv_file)
        #info = csv_file.split(".")[0]  # 这行会导致带.的模型分不出bs_inputlen_outputlen
        info = csv_file.split[:-4]
        all_data_dict[info] = get_json(os.path.join(dataset_path, csv_file))
    
    
    rows = []
    for model_name, kernels in all_data_dict.items():
        first_kernel = True
        for kernel_name, metrics in kernels.items():
            if first_kernel:
                row = {'model_name': model_name, 'kernel_name': kernel_name}
                first_kernel = False
            else:
                row = {'model_name': '', 'kernel_name': kernel_name}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    df.to_excel('model_kernels.xlsx', index=False)



if __name__ == '__main__':
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./mx_profile/"
    main(dataset_path)

