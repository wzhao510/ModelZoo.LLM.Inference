import os
import sys
import onnxruntime as ort
from diffusers import DiffusionPipeline
import importlib
from pathlib import Path
from sd_model.pipeline_onnx_stable_diffusion_3 import OnnxStableDiffusion3Pipeline, OnnxRuntimeModel, FlowMatchEulerDiscreteScheduler
from utils.utils import get_params
import time
from utils.clip_score import ClipScore
import random
import numpy as np

EVAL_MODEL_PATH="/external/ai/models/llm/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

def check_models(params):
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer")):
        print("tokenizer folder No exits, Please check config.json")
        return False
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer_2")):
        print("tokenizer_2 folder No exits, Please check config.json")
        return False
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer_3")):
        print("tokenizer_3 folder No exits, Please check config.json")
        return False
    if not os.path.exists(os.path.join(params["ori_path"], "scheduler")):
        print("scheduler folder No exits, Please check config.json")
        return False

    if not os.path.isfile(os.path.join(params["ori_path"], f"text_encoder/{params['text_encoder']}")):
        print("text_encoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"text_encoder_2/{params['text_encoder_2']}")):
        print("text_encoder_2 Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"text_encoder_3/{params['text_encoder_3']}")):
        print("text_encoder_3 Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"vae_decoder/{params['vae_decoder']}")):
        print("vae_decoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"transformer/{params['transformer']}")):
        print("transformer Model No exits, Please check config.json")
        return False
    return True

def get_gpu_memory_usage(device_id=0):
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

def infer_check_accuracy(sd_text2img_models, step, images_per_prompt, prompts, output_size):
    results = []
    ## infer
    images = sd_text2img_models(prompts, num_inference_steps=step, output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt).images
    
    for idx, image in enumerate(images):
        row = {}
        row["image_name"] = [f"{idx//images_per_prompt:04d}.png"]
        row["prompt"] = prompts[idx//images_per_prompt]
        row["image"] = image
        results.append(row)
    return results


def main(modelname, step=50, images_per_prompt=1, EP="maca", output_size=None, device_id=0):
    if EP.lower() == "maca":
        providers = ["MACAExecutionProvider"]
        init_memory = get_gpu_memory_usage(device_id)
    else:
        providers = ["CPUExecutionProvider"]

    if not os.path.isfile(EVAL_MODEL_PATH):
        raise ValueError(f"{EVAL_MODEL_PATH} dose not exist. Please check on file.")

    params = get_params(modelname)

    # Initialize random seed
    random_int = random.randint(0, 2**32 - 1) if params["seed"] == -1 else params["seed"]
    np.random.seed(random_int)
    check_models(params)

    text_encoder_path = os.path.join(params["ori_path"], f"text_encoder/{params['text_encoder']}")
    text_encoder_2_path = os.path.join(params["ori_path"], f"text_encoder_2/{params['text_encoder_2']}")
    text_encoder_3_path = os.path.join(params["ori_path"], f"text_encoder_3/{params['text_encoder_3']}")
    transformer_path = os.path.join(params["ori_path"], f"transformer/{params['transformer']}")
    vae_decoder_path = os.path.join(params["ori_path"], f"vae_decoder/{params['vae_decoder']}")

    config = DiffusionPipeline.load_config(os.path.join(params["ori_path"], "model_index.json"))

    sess_options = ort.SessionOptions()
    text_encoder_session = ort.InferenceSession(text_encoder_path, providers=providers, sess_options=sess_options, provider_options=[{"keep_model_precision": "1", "device_id": device_id}])
    text_encoder = OnnxRuntimeModel(model=text_encoder_session)
    text_encoder_2_session = ort.InferenceSession(text_encoder_2_path, providers=providers, sess_options=sess_options, provider_options=[{"keep_model_precision": "1", "device_id": device_id}])
    text_encoder_2 = OnnxRuntimeModel(model=text_encoder_2_session)
    text_encoder_3_session = ort.InferenceSession(text_encoder_3_path, providers=providers, sess_options=sess_options, provider_options=[{"keep_model_precision": "1", "device_id": device_id}])
    text_encoder_3 = OnnxRuntimeModel(model=text_encoder_3_session)
    transformer_session = ort.InferenceSession(transformer_path, providers=providers, sess_options=sess_options, provider_options=[{"keep_model_precision": "1", "device_id": device_id}])
    transformer = OnnxRuntimeModel(model=transformer_session)
    vae_decoder_session = ort.InferenceSession(vae_decoder_path, providers=providers, sess_options=sess_options, provider_options=[{"keep_model_precision": "1", "device_id": device_id}])
    vae_decoder = OnnxRuntimeModel(model=vae_decoder_session)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params["ori_path"], subfolder="scheduler")

    new_model_save_dir = Path(params["ori_path"])
    patterns = set(config.keys())
    sub_models_to_load = patterns.intersection({"tokenizer", "tokenizer_2", "tokenizer_3"})
    sub_models = {}
    for name in sub_models_to_load:
        library_name, library_classes = config[name]
        if library_classes is not None:
            library = importlib.import_module(library_name)
            class_obj = getattr(library, library_classes)
            load_method = getattr(class_obj, "from_pretrained")
            # Check if the module is in a subdirectory
            if (new_model_save_dir / name).is_dir():
                sub_models[name] = load_method(new_model_save_dir / name)
            else:
                sub_models[name] = load_method(new_model_save_dir)

    sd_text2img_models = []

    sd_text2img_models.append(
        OnnxStableDiffusion3Pipeline(
            vae_encoder=None,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=sub_models.get("tokenizer", None),
            text_encoder_2=text_encoder_2,
            tokenizer_2=sub_models.get("tokenizer_2", None),
            text_encoder_3=text_encoder_3,
            tokenizer_3=sub_models.get("tokenizer_3", None),
            transformer=transformer,
            scheduler=scheduler,
            feature_extractor=None,
            safety_checker = None,
            requires_safety_checker=False,
        )
    )

    prompt = [
            "a photo of an astronaut riding a horse on mars",
            "A majestic lion jumping from a big stone at night",
            ]

    print("warmup")
    _ = sd_text2img_models[0](prompt, num_inference_steps=step, output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt)

    max_memory = 0.0
    results = []
    print("Start Infer")
    start = time.time()
   
    results = infer_check_accuracy(sd_text2img_models[0], step, images_per_prompt, prompt, output_size)
    end = time.time()
    if EP.lower() == "maca":
        used_memory = get_gpu_memory_usage(0) - init_memory
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
    if EP.lower() == "maca":
        print("StableDiffusion_{}_step{}_images_per_prompt{} FPS : {:.3f}, latency : {:.3f}ms, memory usage: {:.3f} GB".format(model_type, step, images_per_prompt, fps, 1.0/fps*1000, max_memory / 1024 / 1024))
    else:
        print("StableDiffusion_{}_step{}_images_per_prompt{} FPS : {:.3f}, latency : {:.3f}ms".format(model_type, step, images_per_prompt, fps, 1.0/fps*1000))
    print("StableDiffusion_{}_step{}_images_per_prompt{} Avg Score : {:.3f}".format(model_type, step, images_per_prompt, average_score))

if __name__ == '__main__':
    modelname = sys.argv[1]
    step = int(sys.argv[2])
    images_per_prompt = int(sys.argv[3])
    EP = sys.argv[4] if len(sys.argv) > 4 else "maca"
    output_size = int(sys.argv[5]) if len(sys.argv) > 5 else None
    device_id = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    print(f'modelname: {modelname}\nstep: {step}\nimages_per_prompt: {images_per_prompt}\nEP: {EP}\noutput_size: {output_size}\ndevice_id: {device_id}')
    main(modelname, step, images_per_prompt, EP, output_size, device_id)
