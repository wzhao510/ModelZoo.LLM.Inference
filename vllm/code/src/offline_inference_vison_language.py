"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
# from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image


question = "What is the content of this image?"

def inital_LLM(model_path, dtype, enforce_eager, gpu_memory_utilization=0.9, tensor_parallel_size=1, max_num_seqs=5, trust_remote_code=True):  
    llm = LLM(model=model_path, 
              dtype=dtype, 
              enforce_eager=enforce_eager,
              gpu_memory_utilization=gpu_memory_utilization, 
              tensor_parallel_size=tensor_parallel_size,
              max_num_seqs=max_num_seqs,
              trust_remote_code=trust_remote_code)
    return llm

# LLaVA-1.5
def generate_prompt_llava(question, model_path):

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    return prompt


# LLaVA-1.6/LLaVA-NeXT
def generate_prompt_llava_next(question, model_path):

    # prompt = f"[INST] <image>\n{question} [/INST]"
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image?\n ASSISTANT:"
    return prompt


# Fuyu
def generate_prompt_fuyu(question, model_path):

    prompt = f"{question}\n"

    return prompt


# Phi-3-Vision
def generate_prompt_phi3v(question, model_path):

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    # llm = LLM(
    #     model=model_path,
    #     trust_remote_code=True,
    #     max_num_seqs=5,
    # )
    return prompt


# PaliGemma
def generate_prompt_paligemma(question, model_path):

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"

    return prompt


# Chameleon
def generate_prompt_chameleon(question, model_path):

    prompt = f"{question}<image>"
    return prompt


# MiniCPM-V
def generate_prompt_minicpmv(question, model_path):

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt


# InternVL
def generate_prompt_internvl(question, model_path):
    # Generally, InternVL can use chatml template for conversation
    TEMPLATE = "<|im_start|>User\n{prompt}<|im_end|>\n<|im_start|>Assistant\n"
    prompt = f"<image>\n{question}\n"
    prompt = TEMPLATE.format(prompt=prompt)
    # llm = LLM(
    #     model=model_path,
    #     trust_remote_code=True,
    #     max_num_seqs=5,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=2,
    # )
    return prompt


# BLIP-2
def generate_prompt_blip2(question, model_path):

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    return prompt

model_prompt_map = {
    "llava": generate_prompt_llava,
    "llava-next": generate_prompt_llava_next,
    "fuyu": generate_prompt_fuyu,
    "phi3_v": generate_prompt_phi3v,
    "paligemma": generate_prompt_paligemma,
    "chameleon": generate_prompt_chameleon,
    "minicpmv": generate_prompt_minicpmv,
    "blip-2": generate_prompt_blip2,
    "internvl_chat": generate_prompt_internvl,
}

def get_prompt(model_type, question, model_path):
    if model_type not in model_prompt_map:
        raise ValueError(f"Model type {model_type} is not supported.")
    return model_prompt_map[model_type](question, model_path)


def main(args):
    model_type = args.model_type
    model_path = args.model_path
        
    prompt = get_prompt(model_type=model_type, question=question, model_path=model_path)
    llm = inital_LLM(model_path=args.model_path,
                     dtype=args.dtype,
                     enforce_eager=args.enforce_eager,
                     gpu_memory_utilization=args.gpu_memory_utilization,
                     tensor_parallel_size=args.tensor_parallel_size,
                     max_num_seqs=args.max_num_seqs,
                     trust_remote_code=args.trust_remote_code)

    image = Image.open(args.image_path).convert("RGB")
    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.98, max_tokens=512,top_k=1)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        } for _ in range(args.num_prompts)]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    #print(outputs)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(f"generated_text :{generated_text}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models')
    parser.add_argument('--model-type',
                        type=str,
                        required=True,
                        help='model type of vision language')
    parser.add_argument('--model-path',
                        type=str,
                        required=True,
                        help='model path of vision language')
    parser.add_argument('--image-path',
                        type=str,
                        required=True,
                        help='path of input image')
    parser.add_argument('--max-num-seqs',
                        type=int,
                        default=128,
                        help='Number of max seqs.')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')

    args = parser.parse_args()
    main(args)
