import os
import argparse
from utils import get_params

def run_multimodal(args):
    model_config = get_params(args.model)

    model_path = model_config["model_path"]
    model_type = model_config["model_type"]
    tp_size = model_config["tensor_parallel_size"]
    dtype = model_config["dtype"]
    gpu_memory_utilization = model_config["gpu_memory_utilization"]
    max_num_seqs = model_config["max_num_seqs"]

    image_path = "./data/demo.jpeg"
    script = "python ./code/src/offline_inference_vison_language.py"
    if args.enforce_eager:
            run_cmd = "{} --model-type {} --model-path {} --image-path {} --tensor_parallel_size {} --dtype {} --enforce-eager --gpu-memory-utilization {} --max-num-seqs {}".format(
                    script, model_type, model_path, image_path, tp_size, dtype, gpu_memory_utilization, max_num_seqs)
    else:
        run_cmd = "{} --model-type {} --model-path {} --image-path {} --tensor_parallel_size {} --dtype {} --gpu-memory-utilization {} --max-num-seqs {}".format(
                    script, model_type, model_path, image_path, tp_size, dtype, gpu_memory_utilization, max_num_seqs)
    print(run_cmd)    
    os.system(run_cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run vison language demo with offline inference.")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    
    args = parser.parse_args()
    run_multimodal(args)