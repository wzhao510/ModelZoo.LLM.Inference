# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline inference throughput."""

import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Optional, Union

import torch
import uvloop
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from benchmark_dataset import (
    AIMODataset,
    BurstGPTDataset,
    ConversationDataset,
    InstructCoderDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

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

def is_mainstream_model(model_path):
    if model_path.find('Qwen3') > 0:
        return True
    if model_path.find('DeepSeek') > 0:
        return True
    if model_path.find('GLM-4.5') > 0:
        return True
    return False

def run_vllm(
    args,
    llm,
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> tuple[float, Optional[list[RequestOutput]]]:
    from vllm import  SamplingParams
    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of"
        " prompt_len and expected_output_len for all requests."
    )
    # Add the requests to the engine.
    
    
    E2E_TIME = [] # 端到端推理延时
    FIRST_LATENCY = [] # 每个并发首token延时 TTFT
    INFER_LATENCY = [] # 每个并发推理时间延时 ITL
    DECODER_LATENCY = []

    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(
            TokensPrompt(
                prompt_token_ids=request.prompt["prompt_token_ids"],
                multi_modal_data=request.multi_modal_data,
            )
            if "prompt_token_ids" in request.prompt
            else TextPrompt(
                prompt=request.prompt, multi_modal_data=request.multi_modal_data
            )
        )
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )

    # global MX_PROFILE_CSV_NAME
    # model_name_list = args.model.split("/")
    # model_name = model_name_list[-2] if len(model_name_list[-1]) == 0 else model_name_list[-1]
    # MX_PROFILE_CSV_NAME = f"{model_name}_{len(requests)}_{requests[0][1]}_{requests[0][2]}_tp{args.tensor_parallel_size}.csv"

    lora_requests: Optional[list[LoRARequest]] = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    use_beam_search = False

    outputs = None
    if not use_beam_search:
        if args.enable_profile:
            llm.start_profile()
            start = time.perf_counter()
            
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests, use_tqdm=True)
            
            end = time.perf_counter()
            llm.stop_profile()
        else:
            start = time.perf_counter()
            outputs = llm.generate(
                prompts, sampling_params, lora_request=lora_requests, use_tqdm=True
            )
            end = time.perf_counter()

    # E2E_TIME.append(end-start)
    # if not hasattr(output[0], "metrics") is None:
    #     return np.mean(E2E_TIME), None, None

    # for out in output:
    #     ## 打印输出toknen 长度
    #     # print("output token length: ", len((out.outputs[0].token_ids)))
        
    #     if out.metrics.first_token_time is not None:
    #         ## 每个并发首字完成耗时
    #         TTFT = out.metrics.first_token_time - out.metrics.arrival_time
    #         FIRST_LATENCY.append(TTFT)
    #         ## 每个并发Decoding完成耗时
    #         DTL = (out.metrics.finished_time - out.metrics.first_token_time) / (len(out.outputs[0].token_ids) - 1)            
    #         DECODER_LATENCY.append(DTL)

    #     ## 每个并发推理完成耗时
    #     ITL = out.metrics.finished_time - out.metrics.arrival_time
    #     INFER_LATENCY.append(ITL)
    else:
        assert lora_requests is None, "BeamSearch API does not support LoRA"
        prompts = [request.prompt for request in requests]
        # output_len should be the same for all requests.
        output_len = requests[0][2]
        for request in requests:
            assert request.expected_output_len == output_len
        start = time.perf_counter()
        llm.beam_search(
            prompts,
            BeamSearchParams(
                beam_width=n,
                max_tokens=output_len,
                ignore_eos=True,
            ),
        )
        end = time.perf_counter()
    return end - start, outputs


def run_vllm_chat(
    args,
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput]]:
    """
    Run vLLM chat benchmark. This function is recommended ONLY for benchmarking
    multimodal models as it properly handles multimodal inputs and chat
    formatting. For non-multimodal models, use run_vllm() instead.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    start = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


async def run_vllm_async(
    #requests: list[SampleRequest],
    args: argparse.Namespace,
    tokenizer,
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float:

    def post_process(requests, results, elasped_time):

        begin_time = time.time()
        end_time = 0
        
        # if len(requests[0].outputs[0].token_ids) == 128:
        #     import pdb; pdb.set_trace()
        FIRST_LATENCY = [] # 每个并发首token延时 TTFT
        # INFER_LATENCY = [] # 每个并发推理时间延时 ITL
        DECODER_LATENCY = []
        if results[0].metrics is not None:
            for i in results:
                begin_time = min(begin_time, i.metrics.arrival_time)
                end_time = max(end_time, i.metrics.finished_time)
                FIRST_LATENCY.append(i.metrics.first_token_time - i.metrics.arrival_time)
                DTL = (i.metrics.finished_time - i.metrics.first_token_time) / (len(i.outputs[0].token_ids) - 1)            
                DECODER_LATENCY.append(DTL)
                # ITL = i.metrics.finished_time - i.metrics.arrival_time
                # INFER_LATENCY.append(ITL)
                
            elasped_time = end_time - begin_time
            show_result(requests,(elasped_time, np.mean(FIRST_LATENCY), np.mean(DECODER_LATENCY)*1000))
        else:
            show_result(requests, None,elasped_time, None)
            requests = get_requests(
                        args, 
                        tokenizer,
                        num_requests=args.num_prompts,
                        input_len=args.input_len,
                        output_len=args.output_len
                        )
    from vllm import SamplingParams

    async with build_async_engine_client_from_engine_args(
        engine_args, disable_frontend_multiprocessing=disable_frontend_multiprocessing
    ) as llm:
        model_config = await llm.get_model_config()
        print("Start warm up....")
        for idx in range(args.warmup_loops):
            print(f"warm up {idx}...")
            requests = get_requests(
                        args, 
                        tokenizer,
                        num_requests=args.num_prompts,
                        input_len=args.input_len,
                        output_len=args.output_len
                        )
            assert all(
                        model_config.max_model_len
                        >= (request.prompt_len + request.expected_output_len)
                        for request in requests
                    ), (
                        "Please ensure that max_model_len is greater than the sum of"
                        " prompt_len and expected_output_len for all requests."
                    )
            prompts: list[Union[TextPrompt, TokensPrompt]] = []
            sampling_params: list[SamplingParams] = []
            lora_requests: list[Optional[LoRARequest]] = []
            for request in requests:
                prompts.append(
                    TokensPrompt(
                        prompt_token_ids=request.prompt["prompt_token_ids"],
                        multi_modal_data=request.multi_modal_data,
                    )
                    if "prompt_token_ids" in request.prompt
                    else TextPrompt(
                        prompt=request.prompt, multi_modal_data=request.multi_modal_data
                    )
                )
                sampling_params.append(
                    SamplingParams(
                        n=n,
                        temperature=1.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=request.expected_output_len,
                        detokenize=not disable_detokenize,
                    )
                )
                lora_requests.append(request.lora_request)

            generators = []
            start = time.perf_counter()
            for i, (prompt, sp, lr) in enumerate(
                zip(prompts, sampling_params, lora_requests)
            ):
                generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
                generators.append(generator)
            all_gens = merge_async_iterators(*generators)
            results = []
            async for i, res in all_gens:
                results.append(res)
            end = time.perf_counter()
            show_result(args, requests,end - start, None)

        if args.batched_test:
            print("Start batched test....")
            batch_list_normal = [1,8,16,32,64]
            input_len_list = [256, 512, 1024]
            output_len_list = [128, 512, 1024]
            if is_mainstream_model(args.model):
                print("Test with mainstream model")
                batch_list_normal = [1,8,16,32,64,128,256,512,1024]                
                input_len_list = [128, 2048, 3072]
                output_len_list = [128, 2048, 1024]
                if args.highlight_test:
                    print("Test with highlight case")
                    batch_list_normal = [1,16,32,128,512] 
                    input_len_list = [128, 2048, 3072]
                    output_len_list = [128, 2048, 1024]
            print(f"Test {batch_list_normal=}, {input_len_list=}, {output_len_list}")
            for batch in batch_list_normal:
                    for input_len in input_len_list:
                        for output_len in output_len_list:
                            if is_mainstream_model(args.model):
                                if input_len == 128 and output_len != 128:
                                    continue
                                if input_len == 2048 and output_len != 2048:
                                    continue
                                if input_len == 3072 and output_len != 1024:
                                    continue
                                if input_len == 3072 and batch > 64:
                                    continue
                            else:
                                if input_len == 1024 and output_len != 1024:
                                    continue
                            # Add the requests to the engine.
                            requests = get_requests(
                                            args, 
                                            tokenizer,
                                            num_requests=batch,
                                            input_len=input_len,
                                            output_len=output_len
                                            )
                            assert all(
                                    model_config.max_model_len
                                    >= (request.prompt_len + request.expected_output_len)
                                    for request in requests
                                ), (
                                    "Please ensure that max_model_len is greater than the sum of"
                                    " prompt_len and expected_output_len for all requests."
                                )
                            prompts: list[Union[TextPrompt, TokensPrompt]] = []
                            sampling_params: list[SamplingParams] = []
                            lora_requests: list[Optional[LoRARequest]] = []
                            for request in requests:
                                prompts.append(
                                    TokensPrompt(
                                        prompt_token_ids=request.prompt["prompt_token_ids"],
                                        multi_modal_data=request.multi_modal_data,
                                    )
                                    if "prompt_token_ids" in request.prompt
                                    else TextPrompt(
                                        prompt=request.prompt, multi_modal_data=request.multi_modal_data
                                    )
                                )
                                sampling_params.append(
                                    SamplingParams(
                                        n=n,
                                        temperature=1.0,
                                        top_p=1.0,
                                        ignore_eos=True,
                                        max_tokens=request.expected_output_len,
                                        detokenize=not disable_detokenize,
                                    )
                                )
                                lora_requests.append(request.lora_request)

                            generators = []
                            start = time.perf_counter()
                            for i, (prompt, sp, lr) in enumerate(
                                zip(prompts, sampling_params, lora_requests)
                            ):
                                generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
                                generators.append(generator)
                            all_gens = merge_async_iterators(*generators)
                            results = []
                            async for i, res in all_gens:
                                results.append(res)
                            end = time.perf_counter()
                            show_result(args, requests,end - start, None)
        else:
            # Add the requests to the engine.
            requests = get_requests(
                        args, 
                        tokenizer,
                        num_requests=args.num_prompts,
                        input_len=args.input_len,
                        output_len=args.output_len
                        )
            assert all(
                        model_config.max_model_len
                        >= (request.prompt_len + request.expected_output_len)
                        for request in requests
                    ), (
                        "Please ensure that max_model_len is greater than the sum of"
                        " prompt_len and expected_output_len for all requests."
                    )
            prompts: list[Union[TextPrompt, TokensPrompt]] = []
            sampling_params: list[SamplingParams] = []
            lora_requests: list[Optional[LoRARequest]] = []
            for request in requests:
                prompts.append(
                    TokensPrompt(
                        prompt_token_ids=request.prompt["prompt_token_ids"],
                        multi_modal_data=request.multi_modal_data,
                    )
                    if "prompt_token_ids" in request.prompt
                    else TextPrompt(
                        prompt=request.prompt, multi_modal_data=request.multi_modal_data
                    )
                )
                sampling_params.append(
                    SamplingParams(
                        n=n,
                        temperature=1.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=request.expected_output_len,
                        detokenize=not disable_detokenize,
                    )
                )
                lora_requests.append(request.lora_request)
            if args.enable_profile:
                generators = []
                llm.start_profile()
                start = time.perf_counter()
                for i, (prompt, sp, lr) in enumerate(
                    zip(prompts, sampling_params, lora_requests)
                ):
                    generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
                    generators.append(generator)
                all_gens = merge_async_iterators(*generators)
                llm.stop_profile()
            else:
                generators = []
                start = time.perf_counter()
                for i, (prompt, sp, lr) in enumerate(
                    zip(prompts, sampling_params, lora_requests)
                ):
                    generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
                    generators.append(generator)
                all_gens = merge_async_iterators(*generators)
            results = []
            
            async for i, res in all_gens:
                results.append(res)
            end = time.perf_counter()
            show_result(args, requests,end - start, None)


def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
) -> float:
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
    )
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (
                max(max_prompt_len, next_prompt_len)
                + max(max_output_len, next_output_len)
            ) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        if not disable_detokenize:
            # Include the decoding time.
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: list[SampleRequest],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve

    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [request.prompt for request in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k] for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer,num_requests,input_len,output_len):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset."
        )
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs["dataset_split"] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
            common_kwargs["dataset_split"] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return dataset_cls(**common_kwargs).sample(**sample_kwargs)


def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    sync_profile = False
    if args.enable_profile:
        print("[INFO] Seems that you turn on PROFILE. It will slower than normal.")   

        model_name_list = args.model.split("/")
        model_name = model_name_list[-2] if len(model_name_list[-1]) == 0 else model_name_list[-1]
        MX_PROFILE_DIR = f"./mx_vllm_profile/{model_name}_tp{args.tensor_parallel_size}"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = MX_PROFILE_DIR
        if not os.path.exists(MX_PROFILE_DIR):
            os.makedirs(MX_PROFILE_DIR)
        sync_profile = True
    request_outputs: Optional[list[RequestOutput]] = None

    args.compilation_config = CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    
    if args.backend == "vllm":
        if args.async_engine:
             
            elapsed_time = uvloop.run(
                run_vllm_async(
                    args,
                    tokenizer,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                    args.disable_detokenize,
                )
            )
        else:
            from vllm import LLM, SamplingParams

            engine_args = EngineArgs.from_cli_args(args)     
            llm = LLM(**dataclasses.asdict(engine_args))
            # Sample the requests.
            print("Start warm up....")
            args.enable_profile = False
            for idx in range(args.warmup_loops):
                print(f"warm up {idx}...")
                requests = get_requests(
                    args, 
                    tokenizer,
                    num_requests=args.num_prompts,
                    input_len=args.input_len,
                    output_len=args.output_len
                    ) 
                            
                elapsed_time, request_outputs = run_vllm(
                    args,
                    llm,
                    requests,
                    args.n,
                    engine_args,
                    args.disable_detokenize,
                )
                show_result(
                    args=args,
                    requests=requests,
                    elapsed_time=elapsed_time,
                    request_outputs=request_outputs
                    )
            
            args.enable_profile = sync_profile
            if args.batched_test:
                print("Start batched test....")
                for batch in [1,8,16,32,64]:
                    for input_len in [256, 512, 1024]:
                        for output_len in [128, 512, 1024]:
                            if input_len == 1024 and output_len != 1024:
                                continue
                            
                            # Synthesize a prompt with the given input length.
                            requests = get_requests(
                                        args, 
                                        tokenizer,
                                        num_requests=batch,
                                        input_len=input_len,
                                        output_len=output_len
                                        ) 
                            
                            elapsed_time, request_outputs = run_vllm(
                                args,
                                llm,
                                requests,
                                args.n,
                                engine_args,
                                args.disable_detokenize,
                            )
                            show_result(
                                args=args,
                                requests=requests,
                                elapsed_time=elapsed_time,
                                request_outputs=request_outputs
                                )
            
            else:
                
                print("Start performance test....")
                requests = get_requests(
                    args, 
                    tokenizer,
                    num_requests=args.num_prompts,
                    input_len=args.input_len,
                    output_len=args.output_len
                    ) 
                            
                elapsed_time, request_outputs = run_vllm(
                    args,
                    llm,
                    requests,
                    args.n,
                    engine_args,
                    args.disable_detokenize,
                )
                show_result(
                    args=args,
                    requests=requests,
                    elapsed_time=elapsed_time,
                    request_outputs=request_outputs
                    )


            # elapsed_time, request_outputs = run_vllm(
            #     requests,
            #     args.n,
            #     EngineArgs.from_cli_args(args),
            #     args.disable_detokenize,
            # )
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(
            requests,
            args.model,
            tokenizer,
            args.n,
            args.hf_max_batch_size,
            args.trust_remote_code,
            args.disable_detokenize,
        )
    elif args.backend == "mii":
        elapsed_time = run_mii(
            requests, args.model, args.tensor_parallel_size, args.output_len
        )
    elif args.backend == "vllm-chat":
        elapsed_time, request_outputs = run_vllm_chat(
            args,requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    

def show_result(args, requests, elapsed_time, request_outputs=None):
    if request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += (
                len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
            )
            total_output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens
    is_multi_modal = any(request.multi_modal_data is not None for request in requests)
    if is_multi_modal and args.backend != "vllm-chat":
        print(
            "\033[91mWARNING\033[0m: Multi-modal request with "
            f"{args.backend} backend detected. The "
            "following metrics are not accurate because image tokens are not"
            " counted. See vllm-project/vllm/issues/9778 for details."
        )
        # TODO(vllm-project/vllm/issues/9778): Count multi-modal token length.
        # vllm-chat backend counts the image tokens now

    print(
        f"bs_{len(requests)}_input_{requests[0].prompt_len}_output_{requests[0].expected_output_len} Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2,
        )
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    # === Backend Validation ===
    valid_backends = {"vllm", "hf", "mii", "vllm-chat"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if not args.dataset and not args.dataset_path:
        print("When dataset path is not set, it will default to random dataset")
        args.dataset_name = "random"
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
        getattr(args, "hf_subset", None) is not None
        or getattr(args, "hf_split", None) is not None
    ):
        warnings.warn(
            "--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
            stacklevel=2,
        )
    elif args.dataset_name == "hf":
        if args.dataset_path in (
            VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
            | ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm-chat", (
                f"{args.dataset_path} needs to use vllm-chat as the backend."
            )  # noqa: E501
        elif args.dataset_path in (
            InstructCoderDataset.SUPPORTED_DATASET_PATHS
            | AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm", (
                f"{args.dataset_path} needs to use vllm as the backend."
            )  # noqa: E501
        else:
            raise ValueError(f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != "random" and args.random_range_ratio is not None:
        warnings.warn(
            "--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
            stacklevel=2,
        )

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if (
        args.dataset_name not in {"random", "sonnet", None}
        and args.prefix_len is not None
    ):
        warnings.warn(
            "--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
            stacklevel=2,
        )

    # === LoRA Settings ===
    if getattr(args, "enable_lora", False) and args.backend != "vllm":
        raise ValueError("LoRA benchmarking is only supported for vLLM backend")
    if getattr(args, "enable_lora", False) and args.lora_path is None:
        raise ValueError("LoRA path must be provided when enable_lora is True")

    # === Backend-specific Validations ===
    if args.backend == "hf" and args.hf_max_batch_size is None:
        raise ValueError("HF max batch size is required for HF backend")
    if args.backend != "hf" and args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")

    if (
        args.backend in {"hf", "mii"}
        and getattr(args, "quantization", None) is not None
    ):
        raise ValueError("Quantization is only for vLLM backend.")

    if args.backend == "mii" and args.dtype != "auto":
        raise ValueError("dtype must be auto for MII backend.")
    if args.backend == "mii" and args.n != 1:
        raise ValueError("n must be 1 for MII backend.")
    if args.backend == "mii" and args.tokenizer != args.model:
        raise ValueError("Tokenizer must be the same as the model for MII backend.")

    # --data-parallel is not supported currently.
    # https://github.com/vllm-project/vllm/issues/16222
    if args.data_parallel_size > 1:
        raise ValueError(
            "Data parallel is not supported in offline benchmark, \
            please use benchmark serving instead"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf", "mii", "vllm-chat"],
        default="vllm",
    )
    parser.add_argument(
        "--batched-test",
        action="store_true",
        help="Test multiple case but load model once."
    )
    parser.add_argument(
        "--highlight-test",
        action="store_true",
        help="Only test highlight case."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random", "sonnet", "burstgpt", "hf"],
        help="Name of the dataset to benchmark on.",
        default="sharegpt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
        "be a json in form of list[dict[..., conversations: "
        "list[dict[..., value: <prompt_or_response>]]]]",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        default=False,
        help="Use vLLM async engine rather than LLM class.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        default=False,
        help="Disable decoupled async engine frontend.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize the response (i.e. do not include "
            "detokenization time in the measurement)"
        ),
    )
    parser.add_argument(
        '--warmup-loops',
        type=int,
        default=1,
        help='warmup loops before performance benchmark')
    parser.add_argument(
        "--enable-profile",
        action='store_true',
        help="enable profile to collect kernel info.")
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the LoRA adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=None,
        help=f"Number of prefix tokens to be used in RandomDataset "
        "and SonnetDataset. For RandomDataset, the total input "
        "length is the sum of prefix-len (default: "
        f"{RandomDataset.DEFAULT_PREFIX_LEN}) and a random context length "
        "sampled from [input_len * (1 - range_ratio), "
        "input_len * (1 + range_ratio)]. For SonnetDataset, "
        f"prefix_len (default: {SonnetDataset.DEFAULT_PREFIX_LEN}) "
        "controls how much of the input is fixed lines versus "
        "random lines, but the total input length remains approximately "
        "input_len tokens.",
    )
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=None,
        help=f"Range ratio (default : {RandomDataset.DEFAULT_RANGE_RATIO}) "
        "for sampling input/output length, "
        "used only for RandomDataset. Must be in the range [0, 1) to "
        "define a symmetric sampling range "
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dtaset
    parser.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    parser.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    validate_args(args)
    main(args)
