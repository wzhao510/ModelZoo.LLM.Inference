# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

# from vllm.assets.image import ImageAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_arg_parser import create_vllm_parser
from PIL import Image


question = "What is the content of this image?"


def inital_LLM(args):
    return LLM(
        model=args.model_path,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        distributed_executor_backend=args.distributed_executor_backend,
        hf_overrides=args.hf_overrides,
    )


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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # messages = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
    # prompt = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )

    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    modality = "image"
    messages = [{
        'role': 'user',
        'content': f'{modality_placeholder[modality]}\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return (prompt, stop_token_ids)

# GLM-4v
def generate_prompt_glm4v(question, model_path):
    prompt = f"<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>\
        {question}<|assistant|>"

    stop_token_ids = [151329, 151336, 151338]
    return (prompt, stop_token_ids)

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


# Qwen2 VL
def generate_prompt_qwen_vl(question, model_path):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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
    "intern_vl": generate_prompt_internvl,
    "qwen_vl": generate_prompt_qwen_vl,
    "glm4v": generate_prompt_glm4v,
}


def get_prompt(model_type, question, model_path):
    if model_type not in model_prompt_map:
        raise ValueError(f"Model type {model_type} is not supported.")
    return model_prompt_map[model_type](question, model_path)


def main(args):
    model_type = args.model_type
    model_path = args.model_path

    prompt = get_prompt(model_type=model_type, question=question, model_path=model_path)
    if model_type == "glm4v":
        args.hf_overrides = {"architectures": ["GLM4VForCausalLM"]}
    else:
        args.hf_overrides = None
    llm = inital_LLM(args)

    image = Image.open(args.image_path).convert("RGB")
    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    if type(prompt) is tuple:
        prompt, stop_token_ids = prompt
        sampling_params = SamplingParams(temperature=0.2,
                                         max_tokens=64,
                                         stop_token_ids=stop_token_ids
                                         )
    else:
        sampling_params = SamplingParams(temperature=0.98, max_tokens=512, top_k=1)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

    else:
        # Batch inference
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
            for _ in range(args.num_prompts)
        ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    # print(outputs)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(f"generated_text :{generated_text}")

def add_custom_args(parser):  
    parser.add_argument(
        "--model-type", type=str, required=True, help="model type of vision language"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="model path of vision language"
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="path of input image"
    )

if __name__ == "__main__":

    parser = create_vllm_parser(  
        description="Demo on using vLLM for offline inference with "
        "vision language models",  
        add_custom_args=add_custom_args
    )  

    args = parser.parse_args()
    main(args)
