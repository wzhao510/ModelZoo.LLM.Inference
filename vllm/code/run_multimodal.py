import argparse
import subprocess

import utils


def run_multimodal(args, other_args):
    model_config = utils.get_params(args.model)

    model_path = model_config["model_path"]
    model_type = model_config["model_type"]
    model_param = model_config["model_param"]
    tp_size = model_param["tensor_parallel_size"]
    dtype = model_param["dtype"]
    gpu_memory_utilization = model_param.get("gpu_memory_utilization", 0.95)
    max_num_seqs = model_param.get("max_num_seqs", 64)
    max_model_len = model_param.get("max_model_len", 4096)

    image_path = "./data/demo.jpeg"
    run_cmd = [
        "python",
        utils.get_modelzoo_vllm_dir() / "code/src/offline_inference_vison_language.py",
        f"--model-type={model_type}",
        f"--model-path={model_path}",
        f"--image-path={image_path}",
        f"--tensor-parallel-size={tp_size}",
        f"--dtype={dtype}",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-num-seqs={max_num_seqs}",
        f"--max-model-len={max_model_len}",
        f"--trust-remote-code",
    ]

    # 这个排在后面会覆盖配置文件中的同名参数
    run_cmd.extend(other_args)
    print(run_cmd)
    subprocess.run(run_cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vison language demo with offline inference."
    )
    parser.add_argument("--model", type=str, required=True)

    args, other_args = parser.parse_known_args()
    run_multimodal(args, other_args)
