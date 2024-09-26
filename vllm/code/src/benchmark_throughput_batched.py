"""Benchmark offline inference throughput."""
import os
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from torch_profile_utils  import profile_to_csv
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

MX_PROFILE_CSV_NAME = "default_1_1_1.csv"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def get_vllm(
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> float:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    enable_lora = False
    if lora_path is not None:
        if os.path.isdir(lora_path):
            enable_lora = True
        else:
            raise ValueError(f"lora path: {lora_path} does not exit")
    llm = LLM(model=model,
              tokenizer=tokenizer,
              quantization=quantization,
              tensor_parallel_size=tensor_parallel_size,
              seed=seed,
              trust_remote_code=trust_remote_code,
              dtype=dtype,
              max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enforce_eager=enforce_eager,
              kv_cache_dtype=kv_cache_dtype,
              device=device,
              enable_prefix_caching=enable_prefix_caching,
              download_dir=download_dir,
              enable_lora=enable_lora)
    return llm

def run_vllm(
    llm,
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    lora_path: Optional[str] = None,
    enable_profile: Optional[bool] = False,
) -> float:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    enable_lora = False
    if lora_path is not None:
        if os.path.isdir(lora_path):
            enable_lora = True
        else:
            raise ValueError(f"lora path: {lora_path} does not exit")
            
    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        if not enable_lora:
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
            )
        else:
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
                lora_request=LoRARequest("sql_adapter", 1, lora_path)
            )
    
    if not enable_profile:
        start = time.perf_counter()
        # FIXME(woosuk): Do not use internal method.
        llm._run_engine(use_tqdm=True)
        end = time.perf_counter()
        return end - start
    
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,],on_trace_ready=trace_handler_f) as p:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,]) as prof:
        start = time.perf_counter()
        # FIXME(woosuk): Do not use internal method.
        llm._run_engine(use_tqdm=True)
        end = time.perf_counter()

    profile_to_csv(prof, MX_PROFILE_CSV_NAME)
    return end - start


def main(args: argparse.Namespace):
    print(args)
    print("[INFO] Use Batched to run 35 case")
    if args.enable_profile:
        print("[INFO] Seems that you turn on PROFILE. It will slower than normal.")
    random.seed(args.seed)
    llm = get_vllm(args.model, args.tokenizer,
                    args.quantization, args.tensor_parallel_size,
                    args.seed, args.n, args.use_beam_search,
                    args.trust_remote_code, args.dtype,
                    args.max_model_len, args.enforce_eager,
                    args.kv_cache_dtype, args.device,
                    args.enable_prefix_caching,
                    args.gpu_memory_utilization, args.download_dir,
                    args.lora_path)


    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    
    for batch in [1,8,16,32,64]:
        for input_len in [256, 512, 1024]:
            for output_len in [128, 512, 1024]:
                if input_len == 1024 and output_len != 1024:
                    continue
                
                global MX_PROFILE_CSV_NAME
                model_name_list = args.model.split("/")
                model_name = model_name_list[-2] if len(model_name_list[-1]) == 0 else model_name_list[-1]
                MX_PROFILE_CSV_NAME = f"{model_name}_{batch}_{input_len}_{output_len}.csv"
                enable_profile = False
                if args.enable_profile:
                    if input_len == 256 and output_len == 128:
                         enable_profile= True
                    if input_len == 1024 and output_len == 1024:
                        enable_profile= True
                # Synthesize a prompt with the given input length.
                prompt = "hi " * (input_len - 1)
                requests = [(prompt, input_len, output_len)
                            for _ in range(batch)]
                elapsed_time = run_vllm(llm, requests, args.model, args.tokenizer,
                                        args.quantization, args.tensor_parallel_size,
                                        args.seed, args.n, args.use_beam_search,
                                        args.trust_remote_code, args.dtype,
                                        args.max_model_len, args.enforce_eager,
                                        args.kv_cache_dtype, args.device,
                                        args.enable_prefix_caching,
                                        args.gpu_memory_utilization, args.download_dir,
                                        args.lora_path, enable_profile)
                
                total_num_tokens = sum(prompt_len + output_len
                                    for _, prompt_len, output_len in requests)
                print(f"bs_{batch}_input_{input_len}_output_{output_len} Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
                    f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
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
    parser.add_argument("--enforce-eager",
                        type=str2bool,
                        default=True,
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument('--lora_path',
                        type=str,
                        default=None,
                        help='directory to lora model path')
    parser.add_argument(
        "--enable-profile",
        action='store_true',
        help="enable profile to collect kernel info.")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
