import os
import re
import argparse
import dataclasses
from typing import Optional
from utils import get_params
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 16
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    seed: int = 1
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    profile: bool = False
    skip_warmup: bool = False
    do_not_exit: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, "
            "used only for random dataset.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per group of shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help="Disable ignore EOS token",
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=BenchArgs.extra_request_body,
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--do-not-exit",
            action="store_true",
            help="Do not exit the program. This is useful for nsys profile with --duration and --delay.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def get_bench_serving_args(command):
    input_len_match = re.search(r"--random-input-len\s+(\d+)", command)
    output_len_match = re.search(r"--random-output-len\s+(\d+)", command)
    num_prompt_match = re.search(r"--num-prompts\s+(\d+)", command)
    input_len = input_len_match.group(1) if input_len_match else "0"
    output_len = output_len_match.group(1) if output_len_match else "0"
    num_prompt = num_prompt_match.group(1) if num_prompt_match else "0"
    result = f"In{input_len}-out{output_len}-bs{num_prompt}"
    print(result)
    return result


def get_launch_server_args(command):
    output_string = ""
    output_string += "-TCON" if re.search(r"--enable-torch-compile+", command) else "-TCOFF"
    output_string += "-NEXTNON" if re.search(r'--speculative-algo\s+NEXTN', command) else "-NEXTNOFF"
    output_string += "-CGOFF" if re.search(r"--disable-cuda-graph+", command) else "-CGON"

    if re.search(r"--attention-backend\s+(\S+)", command):
        output_string += "-" + re.search(r"--attention-backend\s+(\S+)", command).group(1)
    if re.search(r"--enable-flashinfer-mla", command):
        output_string += "-flashmla"
    
    output_string += "-epmoe" if re.search(r"--enable-ep-moe", command) else ""
    output_string += "-dpatt" if re.search(r"--enable-dp-attention", command) else ""

    tp_size_match = re.search(r"--tp\s+(\d+)",command)
    ep_size_match = re.search(r"--ep\s+(\d+)",command)
    dp_size_match = re.search(r"--dp\s+(\d+)",command)
    if tp_size_match:
        output_string += f"-tp{tp_size_match.group(1)}" 
    if ep_size_match:
        output_string += f"-ep{ep_size_match.group(1)}" 
    if dp_size_match:
        output_string += f"-dp{dp_size_match.group(1)}" 
    
    return output_string


def run_benchmark(args):
    model_config = get_params(args.model_path)
    benchmark_cmd = f'python3 -m sglang.bench_offline_throughput'
    ceval_param = model_config["c-eval_param"]
  
    benchmark_cmd += f' --model-path {model_config["model_path"]}'
    if "dtype" in ceval_param:
        benchmark_cmd += f' --dtype {ceval_param["dtype"]}'

    if "mem-fraction-static" in ceval_param:
        benchmark_cmd += f' --mem-fraction-static {ceval_param["mem-fraction-static"]}'

    if "disable_cuda_graph" in ceval_param and ceval_param["disable_cuda_graph"] == "True":
        benchmark_cmd += " --disable-cuda-graph"

    if "trust_remote_code" in ceval_param and ceval_param["trust_remote_code"] == "True":
        benchmark_cmd += " --trust-remote-code"

    if "tp" in ceval_param:
        benchmark_cmd += f' --tp {ceval_param["tp"]}'

    if args.enable_ep_moe:
        benchmark_cmd += " --enable-ep-moe"
    elif "enable_ep_moe" in ceval_param and ceval_param["enable_ep_moe"] == "True":
        benchmark_cmd += " --enable-ep-moe"
    
    if args.ep_size:
        benchmark_cmd += f' --ep-size {args.ep_size}'
    if "ep_size" in ceval_param:
        benchmark_cmd += f' --ep-size {ceval_param["ep_size"]}'
    
    if args.enable_dp_attention:
        benchmark_cmd += " --enable-dp-attention"
    elif "enable_dp_attention" in ceval_param and ceval_param["enable_dp_attention"] == "True":
        benchmark_cmd += " --enable-dp-attention"
    
    if args.dp_size:
        benchmark_cmd += f' --dp-size {args.dp_size}'
    elif "dp_size" in ceval_param:
        benchmark_cmd += f' --dp-size {ceval_param["dp_size"]}'

    if "dataset-name" in ceval_param:
        benchmark_cmd += f' --dataset-name {ceval_param["dataset-name"]}'

    if args.dataset_path:
        benchmark_cmd += f' --dataset-path {args.dataset_path}'
    elif "dataset-path" in ceval_param:
        benchmark_cmd += f' --dataset-path {ceval_param["dataset-path"]}'
    
    if "random_range_ratio" in ceval_param:
        benchmark_cmd += f' --random-range-ratio {ceval_param["random_range_ratio"]}'

    if args.random_input_len:
        benchmark_cmd += f' --random-input-len {args.random_input_len}'

    if args.random_output_len:
        benchmark_cmd += f' --random-output-len {args.random_output_len}'

    if args.num_prompts:
        benchmark_cmd += f' --num-prompts {args.num_prompts}'

    if args.batched_test:
        benchmark_cmd += " --batched-test"  

    if args.result_filename:
        file_server_args = get_launch_server_args(benchmark_cmd)
        file_bench_args = get_bench_serving_args(benchmark_cmd)
        model_name = args.model_path.split('/')[-2] if args.model_path[-1]=='/' else args.model_path.split('/')[-2]
        benchmark_cmd += f' --result-filename ./result/{model_name}{file_server_args}_{file_bench_args}_{args.result_filename}'

    print(benchmark_cmd)
    
    os.system(benchmark_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    parser.add_argument("--batched-test",
                        action="store_true",
                        help="Test 35 case but load model once.")
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    run_benchmark(args)