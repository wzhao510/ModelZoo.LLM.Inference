import os
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
import sys
from vllm_arg_parser import create_vllm_parser

os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

def run_ngrams(args):
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(
            model=args.model,
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

def run_eagle(args):
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0, top_p=0.95)
        llm = LLM(
            model=args.model,
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

def run_draft(args):
    try:
        prompts = ["144的平方根是多少?请直接给出结果不用解释。"]
        sampling_params = SamplingParams(temperature=0, top_p=0.95)
        llm = LLM(
            model=args.model,
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


def add_custom_args(parser):  
    parser.add_argument('--draft-model-path', type=str)

if __name__ == "__main__":

    parser = create_vllm_parser(  
        description="mtp test",  
        add_custom_args=add_custom_args  
    )
    args = parser.parse_args()
    result = run_ngrams(args) + run_eagle(args) + run_draft(args)
    print("=================================")
    print(f"正确率: {result}/3 ({(result/3)*100:.2f}%)")