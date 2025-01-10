import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from torch_profile_utils import profile_to_csv
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils import set_gpu

def run(args):
    print(args)
    os.environ['MACA_SMALL_PAGESIZE_ENABLE'] = "1"
    model_name_list = args.model_path.split("/")
    model_name = model_name_list[-2] if len(
        model_name_list[-1]) == 0 else model_name_list[-1]
    MX_PROFILE_CSV_NAME = f"{model_name}_{args.num_prompts}_{args.input_len}_{args.output_len}.csv"

    # 加载模型对应的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,device_map="auto",trust_remote_code=True)
    # 设置GPU卡
    set_gpu(tp=args.tensor_parallel_size)
    # 加载预训练模型，设置device_map让模型自动分配到可用的设备（GPU或CPU）上
    model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=args.dtype,device_map="auto",trust_remote_code=True)

    # 设置生成文本的参数
    generation_config = model.generation_config
    generation_config.do_sample = args.do_sample        # 采用贪婪采样，每次生成固定输出
    generation_config.max_new_tokens = args.output_len  # 控制固定生成长度
    generation_config.min_new_tokens = args.output_len  # 控制固定生成长度
    generation_config.temperature = args.temperature    # 控制生成的随机性，值越高越随机
    generation_config.top_p = args.top_p                # 用于sampling，控制生成的多样性
    generation_config.top_k = args.top_k                # 用于sampling，控制生成的多样性

    # 设置输入prompt
    prompt = "hi " * (args.input_len - 1)
    requests = [(prompt)
                for _ in range(args.num_prompts)]
    
    # 对提示文本进行编码
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(requests, padding = True,return_tensors="pt").to("cuda")

    # 进行推理生成文本
    if args.enable_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,]) as prof:
            start = time.perf_counter()
            outputs = model.generate(**input_ids, generation_config=generation_config)
            end = time.perf_counter()
        profile_to_csv(prof, MX_PROFILE_CSV_NAME)
        elapsed_time = end - start
    else:
        start = time.perf_counter()
        outputs = model.generate(**input_ids, generation_config=generation_config)
        end = time.perf_counter()
        elapsed_time = end - start

    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")

    # 对生成的输出进行解码并打印
    # for output in outputs:
    #     print(len(output))
    #     generated_text = tokenizer.decode(output, skip_special_tokens=True)
    #     print(generated_text)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,default="/pde_ai/models/llm/Qwen/Qwen2.5-7B-Instruct/")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument('--trust_remote_code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--input-len", type=int, default=128,help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=128,help="Output length for each request.")
    parser.add_argument("--num-prompts", type=int, default=1,help='Batch size')
    # parser.add_argument(
    #     '--max-model-len',
    #     type=int,
    #     default=None,
    #     help='Maximum length of a sequence (including prompt and output). '
    #     'If None, will be derived from the model.')
    # 采样相关参数
    parser.add_argument(
        '--do-sample',
        type=int,
        default=False)
    parser.add_argument(
        '--temperature',
        type=int,
        default=0.000001)
    parser.add_argument(
        '--top-k',
        type=int,
        default=10)
    parser.add_argument(
        '--top-p',
        type=int,
        default=1)
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='Maximum length of a sequence generate.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-profile",
        action='store_true',
        help="enable profile to collect kernel info.")
    args = parser.parse_args()
    run(args)