import os
import sys
import torch
from diffusers import StableDiffusion3Pipeline
from utils.utils import get_params
import time
from utils.clip_score import ClipScore

EVAL_MODEL_PATH="/external/ai/models/llm/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

def get_gpu_memory_usage(device_id=0):
    """
    Get GPU memory usage for MACA devices
    """
    import subprocess
    memory_cmd = f'mx-smi --show-memory -i {device_id}'
    ret = subprocess.run(memory_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    start_pos = ret.stdout.find('vis_vram used')
    end_pos = ret.stdout.find('vis_vram usage')
    memory_str = ret.stdout[start_pos+13:end_pos-2]
    start_pos = memory_str.find(':')
    end_pos = memory_str.find('KB')
    used_memory = int(memory_str[start_pos+1:end_pos])
    return used_memory

def infer_check_accuracy(pipe, step, images_per_prompt, prompts, output_size):
    """
    Run inference and prepare results for accuracy evaluation
    """
    results = []
    # Run inference
    images = pipe(prompts, num_inference_steps=step, generator=torch.manual_seed(0), output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt).images

    for idx, image in enumerate(images):
        row = {}
        row["image_name"] = [f"{idx//images_per_prompt:04d}.png"]
        row["prompt"] = prompts[idx//images_per_prompt]
        row["image"] = image
        results.append(row)
    return results

def main(modelname, step=50, images_per_prompt=1, output_size=None, device_id=0):
    """
    Main function for SD3.5 Medium inference testing
    """
    if not os.path.isfile(EVAL_MODEL_PATH):
        raise ValueError(f"{EVAL_MODEL_PATH} dose not exist. Please check on file.")

    # Load model configuration
    params = get_params(modelname)
    # Use local model path from config if available, otherwise use the provided modelpath
    model_path = params.get('ori_path', modelname)
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} not found")

    if output_size is None:
        output_size = params.get('output_size', 1024)

    print(f"Loading SD3.5 Medium model from: {model_path}")
    print(f"Steps: {step}, Images per prompt: {images_per_prompt}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Device ID: {device_id}")

    # Set default GPU device using torch.cuda.set_device()
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        raise RuntimeError("CUDA is not available")

    # Initialize memory tracking for MACA
    init_memory = get_gpu_memory_usage(device_id)

    # Load the model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Move model to the specified GPU device
    pipe = pipe.to(f"cuda:{device_id}")

    # Test prompts
    prompt = [
        "a photo of an astronaut riding a horse on mars",
        "A majestic lion jumping from a big stone at night",
    ]

    print("warmup")
    _ = pipe(prompt, num_inference_steps=step, output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt)
    # _ = pipe(prompt, num_inference_steps=step, height=output_size, width=output_size, num_images_per_prompt=images_per_prompt)

    max_memory = 0.0
    results = []
    print("Start Infer")
    start = time.time()

    results = infer_check_accuracy(pipe, step, images_per_prompt, prompt, output_size)
    end = time.time()

    used_memory = get_gpu_memory_usage(device_id) - init_memory
    if used_memory > max_memory:
        max_memory = used_memory

    print(f"Cost time: {end-start}")
    fps = len(prompt[:])*images_per_prompt / (end-start)
    print(results)
    print("Infer Finished...")

    print("Start Save images....")
    # save generated images
    save_image=True
    if save_image:
        cnt = 0
        for sd_pipe_output in results:
            sd_pipe_output["image"].save(f"generated_image_{cnt}.png")
            cnt += 1

    print("Start Eval...")

    clip_score = ClipScore(model_name="ViT-H-14",
                        model_weights_path=EVAL_MODEL_PATH,
                        device="cpu")
    average_score = clip_score.process(results)

    path_list = modelname.split("/")
    if len(path_list[-1]) != 0:
        model_type = path_list[-1]
    else:
        model_type = path_list[-2]

    print(f"Output {len(prompt)*images_per_prompt} images, inference cost {end-start:.3f} seconds")
    print("StableDiffusion_{}_step{}_images_per_prompt{} FPS : {:.3f}, latency : {:.3f}ms, memory usage: {:.3f} GB".format(model_type, step, images_per_prompt, fps, 1.0/fps*1000, max_memory / 1024 / 1024))
    print("StableDiffusion_{}_step{}_images_per_prompt{} Avg Score : {:.3f}".format(model_type, step, images_per_prompt, average_score))

if __name__ == '__main__':
    modelpath = sys.argv[1]
    step = int(sys.argv[2])
    images_per_prompt = int(sys.argv[3])
    output_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    device_id = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    print(f'modelpath: {modelpath}\nstep: {step}\nimages_per_prompt: {images_per_prompt}\noutput_size: {output_size}\ndevice_id: {device_id}')
    main(modelpath, step, images_per_prompt, output_size, device_id)
