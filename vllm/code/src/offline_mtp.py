import os
from vllm import LLM, SamplingParams
import argparse
from vllm.config import CompilationConfig
import sys


os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

def run_ngrams():
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend=args.distributed_executor_backend,
            max_model_len=1024,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 4,
            },
        )
        outputs = llm.generate(prompts, sampling_params)
        out = " "
        print("======== ngrams 测试结果 ========")
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
            out += generated_text 
        if "12" in out:
            print("测试成功，精度正常\n")
            print("=================================")
            return 1
        else:
            print("精度可能出现异常，请手动检查\n")
            print("=================================")
            return 0
    except Exception as e:
        print(f"run_ngrams 运行错误: {e}", file=sys.stderr)
        sys.exit(1) 

def run_eagle():
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0, top_p=0.95)
        llm = LLM(
            model=args.model_path,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend=args.distributed_executor_backend,
            max_model_len=1024,
            speculative_config={
                "model": args.draft_model_path,
                "draft_tensor_parallel_size": 1,
                "num_speculative_tokens":1,
                "method": "eagle3",
            },
        )
        outputs = llm.generate(prompts, sampling_params)
        out = " "
        print("======== eagle3 测试结果 ========")
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
            out += generated_text 
        if "12" in out:
            print("测试成功，精度正常\n")
            print("=================================")
            return 1
        else:
            print("精度可能出现异常，请手动检查\n")
            print("=================================")
            return 0
    except Exception as e:
        print(f"run_eagle3 运行错误: {e}", file=sys.stderr)
        sys.exit(2) 

def run_draft():
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0, top_p=0.95)
        llm = LLM(
            model=args.model_path,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend=args.distributed_executor_backend,
            max_model_len=1024,
            speculative_config={
                "model": args.draft_model_path,
                "num_speculative_tokens": 1,
            },
        )
        outputs = llm.generate(prompts, sampling_params)
        out = " "
        print("======== draft 测试结果 ========")
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
            out += generated_text 
        if "12" in out:
            print("测试成功，精度正常\n")
            print("=================================")
            return 1
        else:
            print("精度可能出现异常，请手动检查\n")
            print("=================================")
            return 0
    except Exception as e:
        print(f"run_draft 运行错误: {e}", file=sys.stderr)
        sys.exit(3) 


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--tensor_parallel_size', type=int,default=1)
parser.add_argument('--distributed_executor_backend',default="ray")
parser.add_argument('--enforce_eager', type = bool, default=True)
parser.add_argument('--draft_model_path', type=str)
parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.85,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.85.')

args = parser.parse_args()
print(args)
if __name__ == "__main__":
    result = run_ngrams() + run_eagle() + run_draft()
    print("=================================")
    print(f"正确率: {result}/3 ({(result/3)*100:.2f}%)")