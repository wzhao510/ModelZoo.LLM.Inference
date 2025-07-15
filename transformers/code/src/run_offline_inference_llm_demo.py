import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils import set_gpu

def run(args):
    os.environ['MACA_SMALL_PAGESIZE_ENABLE'] = "1"
    # 加载模型对应的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,device_map="auto")
    # 设置GPU卡
    set_gpu(tp=args.tensor_parallel_size)
    # 加载预训练模型，设置device_map让模型自动分配到可用的设备（GPU或CPU）上
    model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=args.dtype,device_map="auto")

    # 设置生成文本的参数
    generation_config = model.generation_config
    generation_config.max_new_tokens = args.max_new_tokens  # 生成的最大新token数量，可根据需求调整
    generation_config.temperature = args.temperature  # 控制生成的随机性，值越高越随机
    generation_config.top_p = args.top_p  # 用于sampling，控制生成的多样性
    generation_config.top_k = args.top_k  # 用于sampling，控制生成的多样性

    # 设置输入prompt
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # 对提示文本进行编码
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompts, padding = True,return_tensors="pt").to("cuda")

    # 进行推理生成文本
    outputs = model.generate(**input_ids, generation_config=generation_config)
    # outputs = model.generate(**input_ids, max_length=50)
    # 对生成的输出进行解码并打印
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(generated_text)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,default="/external/ai/models/llm/Qwen/Qwen2.5-7B-Instruct/")
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
    # parser.add_argument(
    #     '--max-model-len',
    #     type=int,
    #     default=None,
    #     help='Maximum length of a sequence (including prompt and output). '
    #     'If None, will be derived from the model.')
    # 采样相关参数
    parser.add_argument(
        '--temperature',
        type=int,
        default=0.8)
    parser.add_argument(
        '--top-k',
        type=int,
        default=10)
    parser.add_argument(
        '--top-p',
        type=int,
        default=0.95)
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='Maximum length of a sequence generate.')
    args = parser.parse_args()
    run(args)