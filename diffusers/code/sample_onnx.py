import os
import sys
import numpy as np
from utils.utils import get_params, convertModel, get_input_shape_info
from utils.clip_score import ClipScore
from sd_model.text2img import Text2ImgModelSess
import time
import onnxruntime as ort
from tqdm import trange

EVAL_MODEL_PATH="/pde_ai/models/llm/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

def check_fp32_models(params):
    if not os.path.exists(os.path.join(params["ori_path"], "tokenizer")):
        print("FP32 tokenizer folder No exits, Please check config.json")
        return False
    if not os.path.exists(os.path.join(params["ori_path"], "scheduler")):
        print("FP32 scheduler folder No exits, Please check config.json")
        return False
    
    if not os.path.isfile(os.path.join(params["ori_path"], f"text_encoder/{params['text_encoder']}")):
        print("FP32 text_encoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"vae_decoder/{params['vae_decoder']}")):
        print("FP32 vae_decoder Model No exits, Please check config.json")
        return False
    if not os.path.isfile(os.path.join(params["ori_path"], f"unet/{params['unet']}")):
        print("FP32 unet Model No exits, Please check config.json")
        return False
    return True

def check_fp16_cfg_file(params):
    # tokenizer
    tokenizer_path = os.path.join(params["fp16_path"], "tokenizer")
    if not os.path.exists(tokenizer_path):
        if not os.path.exists(os.path.join(params["ori_path"], "tokenizer")):
            raise ValueError("tokenizer is missing. Plese check path")
        os.makedirs(tokenizer_path)
        print(f'Try cp -r from FP32 path:{params["ori_path"]}')
        cmd = f"cp -r {params['ori_path']}/tokenizer {params['fp16_path']}"
        os.system(cmd)
    if "static_convert" not in params:
        params["static_convert"] = False
    if "dynamic_batch" not in params:
        params["dynamic_batch"] = False

    # scheduler
    scheduler_path = os.path.join(params["fp16_path"], "scheduler")
    if not os.path.exists(scheduler_path):
        if not os.path.exists(os.path.join(params["ori_path"], "scheduler")):
            raise ValueError("scheduler is missing. Plese check path")
        print(f'Try cp -r from FP32 path:{params["ori_path"]}')
        os.makedirs(scheduler_path)
        cmd = f"cp -r {params['ori_path']}/scheduler {params['fp16_path']}"
        os.system(cmd)

def convert_fp16_model(params, module_name, batchsize, skip_fp16_convert=False):

    fp16_path = os.path.join(params["fp16_path"], module_name)
    fp32_path = os.path.join(params["ori_path"], module_name)
    if not os.path.exists(fp16_path):
        print(f"FP16 path:{fp16_path} does not exists, It will create..")
        os.makedirs(fp16_path)

    fp16_modelname = "model_sim_fp16.onnx"
    fp32_modelname = params[module_name]
    fp32_model_path = os.path.join(fp32_path, fp32_modelname)
    fp16_model_path = os.path.join(fp16_path, fp16_modelname)

    if params["static_convert"]:
        img_size = params["outputs_size"].split("#")
        dynamic_batch = params["dynamic_batch"]
        encoder_hidden_dim = params["encoder_hidden_dim"]
        input_shape = get_input_shape_info(module_name, [int(img_size[0]), int(img_size[1])], batchsize, encoder_hidden_dim, dynamic_batch)
        print(f'input shape is {input_shape}')
        print(f'start convert module {module_name}')
        convertModel(fp32_model_path, fp16_model_path, input_shape, dynamic_batch, skip_fp16_convert)
    else:
        if os.path.isfile(fp16_model_path):
            if skip_fp16_convert:
                return fp16_model_path
            print(f"fp16 model existed, overwriting...")
            os.system(f"rm -rf {fp16_model_path}")
        if not os.path.isfile(fp32_model_path):
            raise ValueError(f"Try to converter from fp32..\n \
                    Howerver fp32 path:{fp32_model_path} does not exit" )
        print("fp16 model start convert....")
        cmd = f"python -m maca_converter --model_path {fp32_model_path} --model_type onnx --output {fp16_model_path} --fp32_to_fp16 1"
        if module_name in ["vae_encoder", "vae_decoder"]:
            cmd = cmd + " --fuse_mha 0"
        os.system(cmd)
    return fp16_model_path

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

def infer_check_accuracy(sd_text2img_models, batch_size, test_set, output_size):
    results = []
    for start_idx in trange(0, len(test_set), batch_size):
        end_idx = min(start_idx + batch_size, len(test_set))
        start_idx -= (start_idx + batch_size - end_idx )
        prompts = []
        prompt_str = []
        for i in range(start_idx, end_idx):
            data = test_set[i]
            prompt_str.append(data["prompt"])
            results.append(data)
        prompts = sd_text2img_models.truncated_prompt_embeds(prompt_str)
        prompt_np = np.array(prompts)
        ## infer
        sd_pipe_output = sd_text2img_models(None, height=output_size, width=output_size, prompt_embeds=prompt_np)
        images = sd_pipe_output.images
        if len(images) != end_idx - start_idx:
            print("WARN: DO not match")
        for idx, image in enumerate(images):
            results[start_idx + idx]["image"] = image
    return results

def main(modelname,batchsize,precision, task="normal", modelfile="./",EP="maca", output_size=None, device_id=0, skip_fp16_convert=False, test_round=10):
    if not os.path.isfile(EVAL_MODEL_PATH):
        raise ValueError(f"{EVAL_MODEL_PATH} dose not exist. Please check on file.")
    
    if EP.lower() == "maca":
        providers = [("MACAExecutionProvider",{
            'device_id':device_id,
        }),]
        init_memory = get_gpu_memory_usage(0)
    else:
        providers = ["CPUExecutionProvider",]

    params = get_params(modelname)
    if output_size is not None:
        params["outputs_size"] = "{}#{}".format(output_size, output_size)
    if params.get("do_fp16_convert", False):
        is_succ = check_fp32_models(params)
        if not is_succ:
            raise ValueError("FP32 model is not Good, Plese check config.json")
    
    check_fp16_cfg_file(params)
    if precision == "fp16":
        text_encoder_path = convert_fp16_model(params, "text_encoder", batchsize, skip_fp16_convert)
        unet_path = convert_fp16_model(params, "unet", batchsize, skip_fp16_convert)
        vae_decoder_path = convert_fp16_model(params, "vae_decoder", batchsize, skip_fp16_convert)
    else:
        text_encoder_path = os.path.join(params["ori_path"], params["text_encoder"])
        unet_path = os.path.join(params["ori_path"], params["unet"])
        vae_decoder_path = os.path.join(params["ori_path"], params["vae_decoder"])
    print(text_encoder_path)
    text_encoder_sess = ort.InferenceSession(text_encoder_path, providers=providers)
    unet_sess = ort.InferenceSession(unet_path, providers=providers)
    vae_decoder_sess = ort.InferenceSession(vae_decoder_path, providers=providers)
    

    sd_text2img_models = []

    sd_text2img_models.append(Text2ImgModelSess(params, text_encoder_sess, unet_sess, vae_decoder_sess))
    
    prompt = [
            "a photo of an astronaut riding a horse on mars",
            "A majestic lion jumping from a big stone at night",
            "purple lego dollhouse with a pool and a swing",	
            "black bearded dog with an injured leg wearing a cone",
            "brown white and black white guinea pigs eating parsley handed to them",
            "The Rosetta Stone lying on the ground, covered in snow.",
            "a high-quality photograph of an armadillo playing a bagpipe while standing on one leg",
            "a white robot with a red mohawk painted as graffiti on a red brick wall", 
            "a panda bear playing ping pong using a blue paddle against an ostrich using a red paddle",
            "Anubis wearing sunglasses and sitting astride a hog motorcyle",
            "a photograph of sand with a bucket, lots of scattered shells but no sandpipers",
            "a shiba inu wearing a beret and black turtleneck",
            "a handpalm with leaves growing from it",
            "panda mad scientist mixing sparkling chemicals",
            "a corgi’s head depicted as an explosion of a nebula",
            "a dolphin in an astronaut suit on saturn",
            "a teddy bear on a skateboard in times square",
            "A Big Ben clock towering over the city of London",
            "A Vietnam map showing Ha Long Bay",
            ]
    
    print("warmup")
    sd_pipe_output = sd_text2img_models[0](prompt[:batchsize])

    dataset = []
    for i in range(batchsize):
        row = {}
        row["image_name"] = [f"{i:04d}.png"]
        row["prompt"] = prompt[i]
        row["image"] = None
        dataset.append(row)
   
    max_memory = 0.0
    results = []
    print("Start Infer")
    cost_time=0.0
    for i in range(test_round):
        start = time.time()
        results = infer_check_accuracy(sd_text2img_models[0], int(batchsize), dataset[:], output_size)
        cost_time += time.time() - start
        if EP.lower() == "maca":
            used_memory = get_gpu_memory_usage(0) - init_memory
            if used_memory > max_memory:
                max_memory = used_memory
    
    print(f"Cost time: {cost_time}")
    fps = len(dataset[:]) * test_round / (cost_time)
    #print(results)
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

    if EP.lower() == "maca":
        print("StableDiffusion_{}_bs{}_prec{} FPS : {:.3f}, latency : {:.3f}ms, memory usage: {:.3f} GB".format(model_type, batchsize, precision, fps, 1.0/fps*1000, max_memory / 1024 / 1024))
    else:
        print("StableDiffusion_{}_bs{}_prec{} FPS : {:.3f}, latency : {:.3f}ms".format(model_type, batchsize, precision, fps, 1.0/fps*1000))
    print("StableDiffusion_{}_bs{}_prec{} Avg Score : {:.3f}".format(model_type, batchsize, precision, average_score))


if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = int(sys.argv[2])
    precision = sys.argv[3]

    task = sys.argv[4]      
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    output_size = int(sys.argv[7]) if len(sys.argv) > 7 else None
    skip_fp16_convert = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    test_round = int(sys.argv[9]) if len(sys.argv) > 9 else 10
    device_id = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    if skip_fp16_convert == 0:
        skip_fp16_convert = False
    else:
        skip_fp16_convert = True
    print(f'modelname: {modelname}\nbatchsize: {batchsize}\nprecision: {precision}\ntask: {task}\n'
          f'model_path: {model_path}\nEP: {EP}\noutput_size: {output_size}\ndevice_id: {device_id}\nskip_fp16_convert: {skip_fp16_convert}\ntest_round: {test_round}\n')
    main(modelname,batchsize,precision,task,model_path, EP, output_size, device_id, skip_fp16_convert, test_round) 
