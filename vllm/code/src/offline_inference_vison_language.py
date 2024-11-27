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

# LLaVA-1.5
def run_llava(question, model_path):

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model=model_path)
    return llm, prompt


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question, model_path):

    # prompt = f"[INST] <image>\n{question} [/INST]"
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image?\n ASSISTANT:"
    llm = LLM(model="/pde_ai/models/llm/LLaVa/llava-v1___6-vicuna-13b-hf",gpu_memory_utilization=0.9,tensor_parallel_size=1)

    return llm, prompt


# Fuyu
def run_fuyu(question, model_path):

    prompt = f"{question}\n"
    llm = LLM(model=model_path)

    return llm, prompt


# Phi-3-Vision
def run_phi3v(question, model_path):

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
    )
    return llm, prompt


# PaliGemma
def run_paligemma(question, model_path):

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model=model_path)

    return llm, prompt


# Chameleon
def run_chameleon(question, model_path):

    prompt = f"{question}<image>"
    llm = LLM(model=model_path)
    return llm, prompt


# MiniCPM-V
def run_minicpmv(question, model_path):

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
    )

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt


# InternVL
def run_internvl(question, model_path):
    # Generally, InternVL can use chatml template for conversation
    TEMPLATE = "<|im_start|>User\n{prompt}<|im_end|>\n<|im_start|>Assistant\n"
    prompt = f"<image>\n{question}\n"
    prompt = TEMPLATE.format(prompt=prompt)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )
    return llm, prompt


# BLIP-2
def run_blip2(question, model_path):

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model=model_path)
    return llm, prompt


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "fuyu": run_fuyu,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "internvl_chat": run_internvl,
}


def main(args):
    model_type = args.model_type
    model_path = args.model_path
    if model_type not in model_example_map:
        raise ValueError(f"Model type {model_type} is not supported.")
    
    llm, prompt = model_example_map[model_type](question, model_path)

    
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
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')

    args = parser.parse_args()
    main(args)
