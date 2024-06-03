import argparse
from vllm import LLM, SamplingParams

def run(model_path, tensor_parallel_size, trust_remote_code):
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
    llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size, trust_remote_code=trust_remote_code)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument('--trust_remote_code',
                        action='store_true',
                        help='trust remote code from huggingface')

    args = parser.parse_args()

    run(args.model, args.tensor_parallel_size, args.trust_remote_code)
