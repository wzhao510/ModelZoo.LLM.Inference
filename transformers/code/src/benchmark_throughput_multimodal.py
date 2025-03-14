import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoProcessor,LlavaForConditionalGeneration,AutoModel,GenerationConfig,AutoModelForVision2Seq,AutoModel
import os
import warnings
from PIL import Image
import os
import sys
from torch_profile_utils import profile_to_csv
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils import set_gpu

def get_model_prompt(model_path,question):
    tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto",trust_remote_code=True)

    if "Internvl" in model_path:
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    elif "llava" in model_path:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    elif "Qwen" in model_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": question,
                    }
                ],
            },
        ]
        prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    else:
        print("model_path error")
    return prompt

def run(args):
    print(args)
    os.environ['MACA_SMALL_PAGESIZE_ENABLE'] = "1"
    model_name_list = args.model_path.split("/")
    model_name = model_name_list[-2] if len(
    model_name_list[-1]) == 0 else model_name_list[-1]
    MX_PROFILE_CSV_NAME = f"{model_name}_{args.num_prompts}_{args.input_len}_{args.output_len}.csv"
    # 设置GPU卡
    set_gpu(tp=args.tensor_parallel_size)

    model = AutoModelForVision2Seq.from_pretrained(args.model_path, torch_dtype=args.dtype,device_map="auto",trust_remote_code=True)

    # 加载处理器
    processor = AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True)

    # 设置生成的参数
    generation_config = model.generation_config
    generation_config.max_new_tokens = args.output_len  # 生成的最大新token数量，可根据需求调整
    generation_config.min_new_tokens = args.output_len  # 生成的最小新token数量，可根据需求调整
    generation_config.temperature = args.temperature  # 控制生成的随机性，值越高越随机
    generation_config.top_p = args.top_p  # 用于sampling，控制生成的多样性
    generation_config.top_k = args.top_k  # 用于sampling，控制生成的多样性

    # 设置输入prompt
    prompt = "hi " * (args.input_len - 1)
    
    # # 打开图片
    image_url = "../../data/demo.jpg"
    image = Image.open(image_url).convert("RGB")
    
    # 提问问题
    prompts=[]
    images=[]
    # question = ["图片中有什么物体","图片颜色是什么"]
    for _ in range(args.num_prompts):
        prompts.append(get_model_prompt(args.model_path, prompt))
        images.append(image)

    #使用处理器对图像和问题进行预处理，将其转换为模型能够接受的张量格式
    inputs = processor(images=images, text=prompts, return_tensors="pt").to("cuda")
    
    #预热
    for _ in range(args.warmup_loops):
        outputs = model.generate(**inputs, generation_config=generation_config)

    # 将预处理后的输入传递给模型进行推理，获取生成的回答
    if args.enable_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,]) as prof:
            start = time.perf_counter()
            outputs = model.generate(**inputs, generation_config=generation_config)
            end = time.perf_counter()
        profile_to_csv(prof, MX_PROFILE_CSV_NAME)
        elapsed_time = end - start
    else:
        start = time.perf_counter()
        outputs = model.generate(**inputs, generation_config=generation_config)
        end = time.perf_counter()
        elapsed_time = end - start

    total_num_tokens = (args.output_len) * args.num_prompts
    print(f"Throughput: {len(prompts) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    
    # 对生成的输出进行解码，获取文本形式的回答
    print(outputs.size())
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,default="/pde_ai/models/llm/LLaVa/llava-v1___6-vicuna-13b-hf")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=4)
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
    parser.add_argument("--num-prompts", type=int, default=2,help='Batch size')
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
        '--warmup-loops',
        type=int,
        default=1,
        help='warmup loops before performance benchmark')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-profile",
        action='store_true',
        help="enable profile to collect kernel info.")
    args = parser.parse_args()
    run(args)