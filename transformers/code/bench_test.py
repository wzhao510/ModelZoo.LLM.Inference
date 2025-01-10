import os
import argparse
from utils import get_params

def run_benchmark(model_name, num_prompt, input_len, output_len, tp):

    model_config = get_params(model_name)
    model_path = model_config["model_path"]
    tensor_parallel_size = model_config["param"]["tensor_parallel_size"]
    if tp is not None:
        tensor_parallel_size = tp
    dtype = model_config["param"]["dtype"]

    enable_profile = os.getenv("MX_VLLM_ENABLE_PROFILE", None)
    enable_profile= False if enable_profile is None else True

    c_eval_cmd = f'python ./code/src/benchmark_throughput_llm.py  \
                        --model={model_path}  --num-prompt {num_prompt} --dtype {dtype} \
                        --input-len {input_len} --output-len {output_len} --tensor-parallel-size {tensor_parallel_size} '
    
    if enable_profile:
        c_eval_cmd += " --enable-profile"


    print(c_eval_cmd)
    
    os.system(c_eval_cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=32,
                        help="Number of prompts for throughput test")
    parser.add_argument("--input-len",
                        type=int,
                        default=1024,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=1024,
                        help="Output length for each request.")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=None,
                        help="TP")
    # parser.add_argument("--batched-test",
    #                     action="store_true",
    #                     help="Test 35 case but load model once.")
    args = parser.parse_args()
    run_benchmark(args.model, args.num_prompts, args.input_len, args.output_len, args.tensor_parallel_size)
