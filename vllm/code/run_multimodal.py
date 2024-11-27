import os
import argparse
from utils import get_params

def run_multimodal(args):
    model_config = get_params(args.model)

    model_path = model_config["model_path"]
    model_type = model_config["model_type"]
    image_path = "./data/demo.jpeg"
    run_cmd = f"python ./code/src/offline_inference_vison_language.py --model-type {model_type} --model-path {model_path} --image-path {image_path}"
    
    print(run_cmd)    
    os.system(run_cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run vison language demo with offline inference.")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    run_multimodal(args)