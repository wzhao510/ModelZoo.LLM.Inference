# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm import LLM, SamplingParams
import argparse
from typing import Dict, List, Optional, Sequence, Tuple, Union,Any

def test_SpeculativeDecoding(prompts,model,tps,sp_model,num_sp_t=5):
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(
        model=model,
        tensor_parallel_size=tps,
        speculative_model=sp_model,
        num_speculative_tokens=num_sp_t,
        speculative_draft_tensor_parallel_size=1,
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def test_SpeculativeDecoding_ngram(prompts,model,tps,ngram_max=4,num_sp_t=5):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(
        model=model,
        tensor_parallel_size=tps,
        speculative_model="[ngram]",
        num_speculative_tokens=num_sp_t,
        ngram_prompt_lookup_max=ngram_max
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--prompts",
                        type=str,
                        default="The future of AI is",
                        help="promts")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="")
    parser.add_argument("--speculative-model",
                        type=str,
                        default="",
                        help="speculative model path")
    parser.add_argument("--num-speculative-tokens",
                        type=int,
                        default=5,
                        help="")
    parser.add_argument("--ngram-prompt-lk-max",
                        type=int,
                        default=4,
                        help="ngram_prompt_lookup_max")
    
    parser.add_argument("--ngram",
                        action="store_true",
                        help="using n-grams")
    args = parser.parse_args()
    prompts = args.prompts.split(",") if args.prompts else []
    if args.ngram :
        test_SpeculativeDecoding_ngram(prompts,args.model,args.tensor_parallel_size,args.ngram_prompt_lk_max,args.num_speculative_tokens)
    else :
        test_SpeculativeDecoding(prompts,args.model,args.tensor_parallel_size,args.speculative_model,args.num_speculative_tokens)