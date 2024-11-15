import os
import sys
import onnxruntime as ort
from diffusers import DiffusionPipeline
import importlib
from pathlib import Path
from sd_model.pipeline_sdxl import ORTStableDiffusionXLPipeline
from utils.utils import get_params
import time

def check_models(params):
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer")):
        print("tokenizer folder No exits, Please check config.json")
        return False
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer_2")):
        print("tokenizer_2 folder No exits, Please check config.json")
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
    if not os.path.isfile(os.path.join(params["ori_path"], f"vae_decoder/{params['vae_decoder']}")):
        print("vae_decoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"vae_encoder/{params['vae_encoder']}")):
        print("vae_encoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"unet/{params['unet']}")):
        print("unet Model No exits, Please check config.json")
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

def main(modelname, step=50, images_per_prompt=1, EP="maca", output_size=None, device_id=0):
    if EP.lower() == "maca":
        providers = [("MACAExecutionProvider",{
            'device_id':device_id,
        }),]
        # init_memory = get_gpu_memory_usage(device_id)
    else:
        providers = ["CPUExecutionProvider",]

    params = get_params(modelname)

    check_models(params)

    text_encoder_path = os.path.join(params["ori_path"], f"text_encoder/{params['text_encoder']}")
    unet_path = os.path.join(params["ori_path"], f"unet/{params['unet']}")
    vae_decoder_path = os.path.join(params["ori_path"], f"vae_decoder/{params['vae_decoder']}")
    vae_encoder_path = os.path.join(params["ori_path"], f"vae_encoder/{params['vae_encoder']}")
    text_encoder_2_path = os.path.join(params["ori_path"], f"text_encoder_2/{params['text_encoder_2']}")

    config = DiffusionPipeline.load_config(os.path.join(params["ori_path"], "model_index.json"))

    text_encoder = ort.InferenceSession(text_encoder_path, providers=providers, sess_options=None, provider_options=None)
    unet = ort.InferenceSession(unet_path, providers=providers, sess_options=None, provider_options=None)
    vae_decoder = ort.InferenceSession(vae_decoder_path, providers=providers, sess_options=None, provider_options=None)
    vae_encoder = ort.InferenceSession(vae_encoder_path, providers=providers, sess_options=None, provider_options=None)
    text_encoder_2 = ort.InferenceSession(text_encoder_2_path, providers=providers, sess_options=None, provider_options=None)

    new_model_save_dir = Path(params["ori_path"])
    patterns = set(config.keys())
    sub_models_to_load = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})
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
        ORTStableDiffusionXLPipeline(
            vae_decoder_session=vae_decoder,
            text_encoder_session=text_encoder,
            unet_session=unet,
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            scheduler=sub_models.get("scheduler"),
            feature_extractor=sub_models.get("feature_extractor", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            vae_encoder_session=vae_encoder,
            text_encoder_2_session=text_encoder_2,
        )
    )

    prompt = [
            "a photo of an astronaut riding a horse on mars",
            "A majestic lion jumping from a big stone at night",
            ]

    print("warmup")
    _ = sd_text2img_models[0](prompt, num_inference_steps=step, output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt)

    total_cost = 0.0
    max_memory = 0.0
    print("Start Infer")

    start_time = time.time()
    images = sd_text2img_models[0](prompt, num_inference_steps=step, output_type="pil", height=output_size, width=output_size, num_images_per_prompt=images_per_prompt).images
    total_cost = time.time() - start_time

    print(f"Output {len(images)} images, inference cost {total_cost:.3f} seconds")

    # save generated images
    if False:
        for i in range(len(images)):
            images[i].save(f"generated_image_{i}.png")

if __name__ == '__main__':
    modelname = sys.argv[1]
    step = int(sys.argv[2])
    images_per_prompt = int(sys.argv[3])
    EP = sys.argv[4] if len(sys.argv) > 4 else "maca"
    output_size = int(sys.argv[5]) if len(sys.argv) > 5 else None
    device_id = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    print(f'modelname: {modelname}\nstep: {step}\nimages_per_prompt: {images_per_prompt}\nEP: {EP}\noutput_size: {output_size}\ndevice_id: {device_id}')
    main(modelname, step, images_per_prompt, EP, output_size, device_id)
