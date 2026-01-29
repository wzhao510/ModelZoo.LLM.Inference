# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm import LLM, SamplingParams
from vllm_arg_parser import create_vllm_parser

def run(args):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1)

    # Create an LLM.
    llm = LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size, 
              trust_remote_code=args.trust_remote_code, max_model_len=args.max_model_len, 
              enforce_eager=args.enforce_eager, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_memory_utilization, distributed_executor_backend=args.distributed_executor_backend)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    
    parser = create_vllm_parser(  
        description="offline inference",  
        add_custom_args=None  
    )  

    args = parser.parse_args()
    run(args)