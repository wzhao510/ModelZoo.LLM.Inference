import argparse

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def run(model_path, tensor_parallel_size, trust_remote_code, max_model_len, lora_path):
    # Sample prompts.
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"]
    )

    # Create an LLM. for enable lora
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=trust_remote_code, 
        max_model_len=max_model_len, 
        enable_lora=True)

    # Generate texts from the prompts for LoRA adapter. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, lora_path)
    )

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/pde_ai/models/llm/Llama/Llama-2-7b-hf/")
    parser.add_argument("--lora_path", type=str, default="/AI-DATA/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument('--trust_remote_code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')

    args = parser.parse_args()

    run(args.model, args.tensor_parallel_size, args.trust_remote_code, args.max_model_len, args.lora_path)
