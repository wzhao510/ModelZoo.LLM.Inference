# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import argparse
import inspect
import os
import numpy as np
import sys
import time
import torch

from PIL import Image
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


class BenchData:
    def __str__(self):
        res = []
        for name in filter(lambda x: not x.startswith("_"), dir(self)):
            member = getattr(self, name)
            if inspect.ismethod(member) or inspect.isfunction(member):
                continue
            res.append(f"{name}: {repr(member)}")
        return "{{{}}}".format(", ".join(res))


class VllmBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.image = Image.open(args.image).convert("RGB")
        self.llm = LLM(
            model=args.model,
            tokenizer=args.model,
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
            hf_overrides=args.hf_overrides,
            show_hidden_metrics_for_version=True,
        )

    def make_input(self, in_len, batch, hint=None):
        prompt = {
            "prompt": self.apply_temlate(
                ("hi " * (in_len - 1)) if hint is None else hint
            ),
            "multi_modal_data": {"image": self.image},
        }
        return [prompt for _ in range(batch)]

    def make_params(self, out_len, ignore_eos=True):
        kvargs = {}
        if ignore_eos:
            kvargs = {"min_tokens": out_len}
        return SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=ignore_eos,
            max_tokens=out_len,
            **kvargs,
        )

    def run(self, in_len, out_len, batch_size):
        prompts = self.make_input(in_len, batch_size)
        params = self.make_params(out_len)
        if self.args.enable_profile:
            
            start = time.perf_counter()
            self.llm.start_profile()
            out = self.llm.generate(prompts, params, use_tqdm=True)
            self.llm.stop_profile()
            end = time.perf_counter()
            
        else:
            start = time.perf_counter()
            out = self.llm.generate(prompts, params, use_tqdm=True)
            end = time.perf_counter()

        
        res = self.prase_benchmark_output(out)
        res.time = end - start
        res.input_length = in_len
        res.output_length = out_len
        res.batch_size = batch_size

        qps = batch_size / res.time
        tps = res.tokens_number / res.time
        if out[0].metrics is not None:
            print(
                f"bs_{batch_size}_input_{in_len}_output_{out_len} "
                f"Throughput: {qps:.2f} requests/s, {tps:.2f} tokens/s, "
                f"TTFT is {res.ttft:.3f}ms, Decoder Latency is {res.decoder_latency:.3f}ms"
            )
        else:
            print(
                f"bs_{batch_size}_input_{in_len}_output_{out_len} "
                f"Throughput: {qps:.2f} requests/s, {tps:.2f} tokens/s, "
            )

        # print(f"benchmark result: {res}")
        return res

    def test(self, out_len) -> str:
        prompt = "What is the content of this iamge?"
        print(f'Question: "{prompt}"')
        prompt = self.make_input(0, 1, prompt)
        params = self.make_params(out_len, False)
        out = self.llm.generate(prompt, params, use_tqdm=True)
        return out[0].outputs[0].text

    def prase_benchmark_output(self, data):
        input_tokens_number = 0
        output_tokens_number = 0
        arrival_time = time.time()
        first_scheduled_time = arrival_time
        first_token_time = arrival_time
        finished_time = 0
        # texts = []
        decoder_latency = []
        metrics_is_none = False
        for i in data:
            # texts.append(list(map(lambda x: x.text, i.outputs)))
            input_tokens_number += len(i.prompt_token_ids)
            output_tokens_number += sum(map(lambda x: len(x.token_ids), i.outputs))
            if i.metrics is None:
                metrics_is_none = True
                continue
                
            arrival_time = min(arrival_time, i.metrics.arrival_time)

            first_scheduled_time = min(
                first_scheduled_time, i.metrics.first_scheduled_time
            )
            first_token_time = min(first_token_time, i.metrics.first_token_time)
            finished_time = max(finished_time, i.metrics.finished_time)
            decoder_latency.append(
                (i.metrics.finished_time - i.metrics.first_token_time)
                / (len(i.outputs[0].token_ids) - 1)
            )

        res = BenchData()
        # res.texts = texts
        res.input_tokens_number = input_tokens_number
        res.output_tokens_number = output_tokens_number
        res.tokens_number = input_tokens_number + output_tokens_number

        if not metrics_is_none:
            res.arrival_time = arrival_time
            res.first_scheduled_time = first_scheduled_time
            res.first_token_time = first_token_time
            res.finished_time = finished_time
            res.elapsed_time = finished_time - first_scheduled_time

            res.ttft = (first_token_time - arrival_time) * 1000
            res.tpot = (
                (finished_time - first_token_time)
                * 1000
                / (output_tokens_number - len(data))
            )
            res.decoder_latency = np.mean(decoder_latency) * 1000
            res.tps = res.tokens_number / res.elapsed_time
            res.output_pts = output_tokens_number / res.elapsed_time
        return res

    def apply_temlate(self, prompt: str) -> str:
        template = PromptTemplate()
        try:
            method = getattr(template, self.args.model_type)
        except AttributeError:
            exit(1)
        return method(prompt)


class PromptTemplate:
    def qwen_vl(self, question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def intern_vl(self, question: str) -> str:
        return f"<|im_start|>user\n<image>\n{question}\n<|im_end|>\n<|im_start|>assistant\n"

    def internvl_chat(self, question: str) -> str:
        return self.intern_vl(question)

    def glm4v(self, question: str) -> str:
        return f"<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>{question}<|assistant|>"


def main(args: argparse.Namespace):

    if args.model_type == "glm4v":
        args.hf_overrides = {"architectures": ["GLM4VForCausalLM"]}
    else:
        args.hf_overrides = None

    if args.enable_profile:
        MX_PROFILE_DIR = f"./mx_vllm_profile/{args.model}_tp{args.tensor_parallel_size}_pp{args.pipeline_parallel_size}"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = MX_PROFILE_DIR
        if not os.path.exists(MX_PROFILE_DIR):
            os.makedirs(MX_PROFILE_DIR)
    llm = VllmBenchmark(args)
    print('Answer: "{}"'.format(llm.test(1024)))

    yaml_files = set()

    if args.print_flash_attn_shape:
        os.environ["MACA_LAUNCH_BLOCKING"] = "1"
        os.environ["MHA_PRINT_PARA"] = "ON"
        os.environ["MHA_INPUT_COLLECTION"] = "1"
        yaml_files = set(find_yaml_files("."))

    if args.benchmark_all:
        for batch in [1, 8, 16, 32, 64]:
            for in_len in [256, 512, 1024]:
                for out_len in [128, 512, 1024]:
                    if in_len == 1024 and out_len != 1024:
                        continue
                    llm.run(in_len, out_len, batch)
    else:
        assert args.input_len is not None
        assert args.output_len is not None
        assert args.num_prompts is not None
        llm.run(args.input_len, args.output_len, args.num_prompts)

    if args.print_flash_attn_shape:
        for i, file in enumerate(
            filter(lambda x: x not in yaml_files, find_yaml_files("."))
        ):
            model_name = Path(args.model).name.lower()
            name = f"./{model_name}-in-{args.input_len}-out-{args.output_len}-bs-{args.num_prompts}-flash-attn.yaml"
            Path(file).rename(name)


def find_yaml_files(directory):
    directory = Path(directory)
    yield from directory.glob("*.yaml")
    yield from directory.glob("*.yml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")

    parser.add_argument("--image", type=str, default="./data/demo.jpg")
    parser.add_argument("--print-flash-attn-shape", action="store_true")
    parser.add_argument(
        "--benchmark-all", action="store_true", help="benchmark 35 case"
    )
    parser.add_argument("--model-type", type=str, required=True)

    parser.add_argument("--backend", type=str, choices=["vllm"], default="vllm")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to the dataset."
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
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--quantization", "-q", choices=[*QUANTIZATION_METHODS, None], default=None
    )
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--use-new-beam-search-impl", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum length of a sequence (including prompt and output). "
        "If None, will be derived from the model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="data type for model weights and activations. "
        'The "auto" option will use FP16 precision '
        "for FP32 and FP16 models, and BF16 precision "
        "for BF16 models.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="the fraction of GPU memory to be used for "
        "the model executor, which can range from 0 to 1."
        "If unspecified, will use the default value of 0.9.",
    )
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="device type for vLLM execution",
    )
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.",
    )
    parser.add_argument(
        "--use-v2-block-manager", action="store_true", help="Enable block manager v2."
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable automatic prefix caching for vLLM backend.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="enable chunked prefill for vLLM backend.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="maximum number of batched tokens per " "iteration",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="directory to download and load the weights, "
        "default to the default cache dir of huggingface",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument(
        "--distributed-executor-backend",
        choices=["ray", "mp"],
        default=None,
        help="Backend to use for distributed serving. When more than 1 GPU "
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.',
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        choices=[
            "auto",
            "pt",
            "safetensors",
            "npcache",
            "dummy",
            "tensorizer",
            "bitsandbytes",
        ],
        help="The format of the model weights to load.\n\n"
        '* "auto" will try to load the weights in the safetensors format '
        "and fall back to the pytorch bin format if safetensors format "
        "is not available.\n"
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        "a numpy cache to speed up the loading.\n"
        '* "dummy" will initialize the weights with random values, '
        "which is mainly for profiling.\n"
        '* "tensorizer" will load the weights using tensorizer from '
        "CoreWeave. See the Tensorize vLLM Model script in the Examples"
        "section for more information.\n"
        '* "bitsandbytes" will load the weights using bitsandbytes '
        "quantization.\n",
    )
    parser.add_argument(
        "--disable-async-output-proc",
        action="store_true",
        default=False,
        help="Disable async output processor for vLLM backend.",
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
        "--enable-profile",
        action="store_true",
        help="enable profile to collect kernel info.",
    )

    print(f"sys args: {sys.argv}")
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)
