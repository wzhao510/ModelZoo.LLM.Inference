# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Benchmark offline inference throughput."""
import os
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
from torch_profile_utils  import profile_to_csv

import torch
import numpy as np
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.distributed import cleanup_dist_env_and_memory

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

import asyncio
import uuid

MX_PROFILE_DIR = "./mx_vllm_profile"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
async def run_vllm_async(
    # requests: List[Tuple[str, int, int]],
    args: argparse.Namespace,
    tokenizer
) -> float:
    def post_process(requests, results):
        begin_time = time.time()
        end_time = 0
        for i in results:
            begin_time = min(begin_time, i.metrics.first_scheduled_time)
            end_time = max(end_time, i.metrics.finished_time)

        elasped_time = end_time - begin_time
        show_result(requests, (elasped_time, None, None))

    from vllm.engine.arg_utils import AsyncEngineArgs
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=args.kv_cache_dtype,
            device=args.device,
            enable_prefix_caching=args.enable_prefix_caching,
            download_dir=args.download_dir,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            load_format=args.load_format,
            num_scheduler_steps=args.num_scheduler_steps,
            use_v2_block_manager=args.use_v2_block_manager,
            disable_async_output_proc=args.disable_async_output_proc,
        )
    )

    params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    print("Start warm up....")
    for idx in range(args.warmup_loops):
        print(f"warm up {idx}...")
        requests = prepare_request(args.input_len, args.output_len, args.num_prompts, tokenizer)
        
        tasks = list(map(lambda x: asyncio.create_task(
            vllm_async_generate(engine, x[0], params, uuid.uuid4())), requests))
        res = [await task for task in tasks]
        post_process(requests, res)
        
    if args.batched_test:
        print("Start batched test....")
        for batch in [1,8,16,32,64]:
            for input_len in [256, 512, 1024]:
                for output_len in [128, 512, 1024]:
                    if input_len == 1024 and output_len != 1024:
                        continue

                    # Synthesize a prompt with the given input length.
                    requests = prepare_request(input_len, output_len, batch, tokenizer) 
                    params = SamplingParams(
                        n=args.n,
                        temperature=1.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=args.output_len,
                    )
                    
                    tasks = list(map(lambda x: asyncio.create_task(
                    vllm_async_generate(engine, x[0], params, uuid.uuid4())), requests))
                    res = [await task for task in tasks]
                    post_process(requests, res)
    else:
        print("Start performance test....")
        if not args.enable_profile:
            tasks = list(map(lambda x: asyncio.create_task(
                vllm_async_generate(engine, x[0], params, uuid.uuid4())), requests))
            res = [await task for task in tasks]
        else:
            engine.start_profile()
            tasks = list(map(lambda x: asyncio.create_task(
                vllm_async_generate(engine, x[0], params, uuid.uuid4())), requests))
            res = [await task for task in tasks]
            engine.stop_profile()

        post_process(requests, res)

    engine.shutdown_background_loop()
    del engine
    await asyncio.sleep(0.1)
    cleanup_dist_env_and_memory()

async def vllm_async_generate(engine, prompt, params, id):
    results_generator = engine.generate(prompt, params, id)

    final_output = None

    async for request_output in results_generator:
        final_output = request_output

    return final_output

def get_vllm(args):
    enable_lora = False
    if args.lora_path is not None:
        if os.path.isdir(args.lora_path):
            enable_lora = True
        else:
            raise ValueError(f"lora path: {args.lora_path} does not exit")
    if   args.speculative_model is None:
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=args.kv_cache_dtype,
            device=args.device,
            enable_prefix_caching=args.enable_prefix_caching,
            download_dir=args.download_dir,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            load_format=args.load_format,
            num_scheduler_steps=args.num_scheduler_steps,
            use_v2_block_manager=args.use_v2_block_manager,
            disable_async_output_proc=args.disable_async_output_proc,
            enable_lora=enable_lora,
            disable_sliding_window=args.disable_sliding_window
        )
    elif  "[ngram]" in  args.speculative_model :
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=args.kv_cache_dtype,
            device=args.device,
            enable_prefix_caching=args.enable_prefix_caching,
            download_dir=args.download_dir,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            load_format=args.load_format,
            num_scheduler_steps=args.num_scheduler_steps,
            use_v2_block_manager=args.use_v2_block_manager,
            disable_async_output_proc=args.disable_async_output_proc,
            enable_lora=enable_lora,
            disable_sliding_window=args.disable_sliding_window,
            speculative_model=args.speculative_model,
            num_speculative_tokens=args.num_speculative_tokens,
            ngram_prompt_lookup_max=args.ngram_prompt_lk_max,
        )
    else :
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=args.kv_cache_dtype,
            device=args.device,
            enable_prefix_caching=args.enable_prefix_caching,
            download_dir=args.download_dir,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            distributed_executor_backend=args.distributed_executor_backend,
            load_format=args.load_format,
            num_scheduler_steps=args.num_scheduler_steps,
            use_v2_block_manager=args.use_v2_block_manager,
            disable_async_output_proc=args.disable_async_output_proc,
            enable_lora=enable_lora,
            disable_sliding_window=args.disable_sliding_window,
            speculative_model=args.speculative_model,
            num_speculative_tokens=args.num_speculative_tokens,
            speculative_draft_tensor_parallel_size=1,
        )

    return llm

def run_vllm(
    llm,
    requests: List[Tuple[str, int, int]],
    n: int,
    lora_path: Optional[str] = None,
    enable_profile: Optional[bool] = False,
) -> float:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    enable_lora = False
    if lora_path is not None:
        if os.path.isdir(lora_path):
            enable_lora = True
        else:
            raise ValueError(f"lora path: {lora_path} does not exit")
        
    # Add the requests to the engine.
        
    prompts = []
    sampling_params = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ))
    
    E2E_TIME = [] # 端到端推理延时
    FIRST_LATENCY = [] # 每个并发首token延时 TTFT
    INFER_LATENCY = [] # 每个并发推理时间延时 ITL
    DECODER_LATENCY = []

    if enable_lora:
        if enable_profile:
            start = time.perf_counter()
            llm.start_profile()
            output = llm.generate(prompts, sampling_params, lora_request=LoRARequest("sql_adapter", 1, lora_path), use_tqdm=True)
            llm.stop_profile()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            output = llm.generate(prompts, sampling_params, lora_request=LoRARequest("sql_adapter", 1, lora_path), use_tqdm=True)
            end = time.perf_counter()
    else:
        if enable_profile:            
            start = time.perf_counter()
            llm.start_profile()
            output = llm.generate(prompts, sampling_params, use_tqdm=True)
            llm.stop_profile()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            output = llm.generate(prompts, sampling_params, use_tqdm=True)
            end = time.perf_counter()
    
    E2E_TIME.append(end-start)
    if output[0].metrics is None:
        return np.mean(E2E_TIME), None, None

    for out in output:
        ## 打印输出toknen 长度
        # print("output token length: ", len((out.outputs[0].token_ids)))
        
        if out.metrics.first_token_time is not None:
            ## 每个并发首字完成耗时
            TTFT = out.metrics.first_token_time - out.metrics.arrival_time
            FIRST_LATENCY.append(TTFT)
            ## 每个并发Decoding完成耗时
            DTL = (out.metrics.finished_time - out.metrics.first_token_time) / (len(out.outputs[0].token_ids) - 1)            
            DECODER_LATENCY.append(DTL)

        ## 每个并发推理完成耗时
        ITL = out.metrics.finished_time - out.metrics.arrival_time
        INFER_LATENCY.append(ITL)
    return np.mean(E2E_TIME), np.mean(FIRST_LATENCY), np.mean(DECODER_LATENCY)*1000

def prepare_request(input_len, output_len, num_prompts, tokenizer):
    from pathlib import Path
    with open("{}/../../data/input_data.txt".format(Path(__file__).absolute().parent)) as file:
        lines = [line.strip() for line in file if line.strip()]
        txt_data = " ".join(lines)
    all_tokens = tokenizer(txt_data)
    all_tokens.input_ids = all_tokens.input_ids[:8192]
    token_len = len(all_tokens.input_ids)

    need_gen_req = True
    token_diff = 0
    while need_gen_req:
        offsets = np.random.randint(0, token_len - (input_len - token_diff))
        tmp_tokens = all_tokens.input_ids[offsets : offsets + (input_len - token_diff)]

        prompt = tokenizer.decode(tmp_tokens)
        
        # verify effectiveness
        ver_tokens = tokenizer(prompt).input_ids
        token_diff = len(ver_tokens) - len(tmp_tokens)
        
        if token_diff > 0:
            prompt = tokenizer.decode(tmp_tokens[: -(token_diff)])
            need_gen_req = False
        elif token_diff == 0:
            need_gen_req = False
        else:
            # if the length of re-encoding token list is less than intercapted length,
            # we must re-intercapte the tmp_tokens with new parameter
            print(f"re-tokenized length is less than the first. ver_tokens: {len(ver_tokens)}, tmp_tokens: {len(tmp_tokens)}")
            continue

    requests = [(prompt, input_len, output_len)
                    for _ in range(num_prompts)]
    return requests

def show_result(requests: List, 
                infer_costs: Tuple):
    total_num_tokens = sum(prompt_len + output_len
                            for _, prompt_len, output_len in requests)
    _, prompt_len, output_len = requests[0]
    elapsed_time, ttft, decoder_latency = infer_costs

    if ttft is None and decoder_latency is None:
        print(f"bs_{len(requests)}_input_{prompt_len}_output_{output_len} Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
            f"{total_num_tokens / elapsed_time:.2f} tokens/s")        
    else:
        print(f"bs_{len(requests)}_input_{prompt_len}_output_{output_len} Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
                f"{total_num_tokens / elapsed_time:.2f} tokens/s, "
                f"TTFT is {round(ttft*1000, 2)} ms, Decoder Latency is {round(decoder_latency, 2)} ms")

def main(args: argparse.Namespace):
    print(args)
    
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id,
                    tokenizer_mode='auto',
                    trust_remote_code=args.trust_remote_code)

    if args.enable_profile:
        print("[INFO] Seems that you turn on PROFILE. It will slower than normal.")   

        model_name_list = args.model.split("/")
        model_name = model_name_list[-2] if len(model_name_list[-1]) == 0 else model_name_list[-1]
        MX_PROFILE_DIR = f"./mx_vllm_profile/{model_name}_tp{args.tensor_parallel_size}"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = MX_PROFILE_DIR
        if not os.path.exists(MX_PROFILE_DIR):
            os.makedirs(MX_PROFILE_DIR)

    random.seed(args.seed)
    if not args.async_engine:        
        llm = get_vllm(args)
    
        # Sample the requests.
        print("Start warm up....")
        for idx in range(args.warmup_loops):
            print(f"warm up {idx}...")
            requests = prepare_request(args.input_len, args.output_len, args.num_prompts, tokenizer)
            
            elapsed_time, ttft, decoder_latency = run_vllm(llm, requests, args.n,
                                        args.lora_path)
            infer_costs = (elapsed_time, ttft, decoder_latency)
            show_result(requests, infer_costs)

        if args.batched_test:
            print("Start batched test....")
            for batch in [1,8,16,32,64]:
                for input_len in [256, 512, 1024]:
                    for output_len in [128, 512, 1024]:
                        if input_len == 1024 and output_len != 1024:
                            continue

                        # Synthesize a prompt with the given input length.
                        requests = prepare_request(input_len, output_len, batch, tokenizer) 
                        
                        elapsed_time, ttft, decoder_latency = run_vllm(llm, requests, args.n,
                                                    args.lora_path, args.enable_profile)
                        infer_costs = (elapsed_time, ttft, decoder_latency)
                        show_result(requests, infer_costs)
        
        else:
            print("Start performance test....")
            requests = prepare_request(args.input_len, args.output_len, args.num_prompts, tokenizer)   
            
            elapsed_time, ttft, decoder_latency = run_vllm(llm, requests, args.n,
                                        args.lora_path, args.enable_profile)
            infer_costs = (elapsed_time, ttft, decoder_latency)
            show_result(requests, infer_costs)
    else:
        import asyncio
        asyncio.run(run_vllm_async(args=args, tokenizer=tokenizer))


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
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--use-new-beam-search-impl", action="store_true")
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
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument("--batched-test",
                        action="store_true",
                        help="Test 35 case but load model once.")
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=['cuda', 'cpu', 'auto'],
                        help='device type for vLLM execution')
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument("--use-v2-block-manager",
                        action='store_true',
                        help="Enable block manager v2.")
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="Enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default='auto',
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        "--disable-async-output-proc",
        action='store_true',
        default=False,
        help="Disable async output processor for vLLM backend.")
    parser.add_argument(
        "--disable-sliding-window",
        action='store_true',
        default=False,
        help="Disable sliding window for vLLM backend.")
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument('--lora_path',
                        type=str,
                        default=None,
                        help='directory to lora model path')
    parser.add_argument(
        '--warmup-loops',
        type=int,
        default=1,
        help='warmup loops before performance benchmark')
    parser.add_argument(
        "--enable-profile",
        action='store_true',
        help="enable profile to collect kernel info.")
    
    parser.add_argument("--speculative-model",
                        type=str,
                        default=None,
                        help="speculative model path")
    
    parser.add_argument("--num-speculative-tokens",
                        type=int,
                        default=4,
                        help="")
    parser.add_argument("--ngram-prompt-lk-max",
                        type=int,
                        default=4,
                        help="ngram_prompt_lookup_max")


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
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
