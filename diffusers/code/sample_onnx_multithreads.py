import os
import sys
import csv
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange
import numpy as np
import onnxruntime as ort
import logging


from utils.utils import get_params
from utils.clip_score import ClipScore
from sd_model.text2img import Text2ImgModelSess

EVAL_MODEL_PATH="/PDE-AI/Models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

def split_dataset(image_list, thread_num):
    sub_num = len(image_list)//thread_num
    mod_num = len(image_list)%thread_num
    sub_lists = []
    start_id = 0
    si_list = []
    for i in range(thread_num):
        if i < mod_num:
            sub_lists.append(image_list[start_id:start_id+sub_num+1])
            si_list.append(start_id)
            start_id += (sub_num+1)
        else:
            sub_lists.append(image_list[start_id:start_id+sub_num])
            si_list.append(start_id)
            start_id += sub_num
    return sub_lists, si_list

def check_accuracy_thread(sd_text2img_models, batch_size, test_set, si):
    # print("thread {} start".format(si))
    results = []
    for start_idx in trange(0, len(test_set), batch_size):
        end_idx = min(start_idx + batch_size, len(test_set))

        prompts = []
        for i in range(start_idx, end_idx):
            data = test_set[i]
            prompt = sd_text2img_models.truncated_prompt_embeds(data["prompt"])
            prompts.append(prompt)
            results.append(data)
        
        prompt_np = np.vstack(prompts)
        ## infer
        sd_pipe_output = sd_text2img_models(None, height=512, width=512, prompt_embeds=prompt_np)
        images = sd_pipe_output.images
        if len(images) != end_idx - start_idx:
            print("WARN: DO not match")
        for idx, image in enumerate(images):
            results[start_idx + idx]["image"] = image
    # print("thread {} done".format(si))
    return results

def check_accuracy_multithread(contexts, batch_size, test_set, thread_num=1):
    sub_lists, si_list = split_dataset(test_set, thread_num)
    
    threads_pool = ThreadPoolExecutor(max_workers=thread_num)
    future_list = []
    for i, (sublist, si) in enumerate(zip(sub_lists,si_list)):
        future_list.append(threads_pool.submit(check_accuracy_thread, contexts[i], batch_size, sublist, si))
    
    all_preds = []
    for future in future_list:
        all_preds += future.result()
    print("all threads done")
    
    return all_preds

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

    # scheduler
    scheduler_path = os.path.join(params["fp16_path"], "scheduler")
    if not os.path.exists(scheduler_path):
        if not os.path.exists(os.path.join(params["ori_path"], "scheduler")):
            raise ValueError("scheduler is missing. Plese check path")
        print(f'Try cp -r from FP32 path:{params["ori_path"]}')
        os.makedirs(scheduler_path)
        cmd = f"cp -r {params['ori_path']}/scheduler {params['fp16_path']}"
        os.system(cmd)

def convert_fp16_model(params, module_name):

    fp16_path = os.path.join(params["fp16_path"], module_name)
    fp32_path = os.path.join(params["ori_path"], module_name)
    if not os.path.exists(fp16_path):
        print(f"FP16 path:{fp16_path} does not exists, It will create..")
        os.makedirs(fp16_path)

    fp16_modelname = "model_sim_fp16.onnx"
    fp32_modelname = params[module_name]
    fp32_model_path = os.path.join(fp32_path, fp32_modelname)
    fp16_model_path = os.path.join(fp16_path, fp16_modelname)
    if os.path.isfile(fp16_model_path):
        return fp16_model_path
    if not os.path.isfile(fp32_model_path):
        raise ValueError(f"Try to converter from fp32..\n \
                Howerver fp32 path:{fp32_model_path} does not exit" )
    print("fp16 model start convert....")
    cmd = f"python -m maca_converter --model_path {fp32_model_path} --model_type onnx --output {fp16_model_path} --fp32_to_fp16 1"
    if module_name in ["vae_encoder", "vae_decoder"]:
        cmd = cmd + " --fuse_mha 0"
    os.system(cmd)
    return fp16_model_path


def main(modelname,batchsize,precision, task="normal", modelfile="./",EP="maca", th_num="8"):
    if not os.path.isfile(EVAL_MODEL_PATH):
        raise ValueError(f"{EVAL_MODEL_PATH} dose not exist. Please check on file.")
            
    if EP.lower() == "maca":
        providers = ["MACAExecutionProvider",]
    else:
        providers = ["CPUExecutionProvider",]


    params = get_params(modelname)
    if params.get("do_fp16_convert", False):
        is_succ = check_fp32_models(params)
        if not is_succ:
            raise ValueError("FP32 model is not Good, Plese check config.json")
    
    check_fp16_cfg_file(params)
    if precision == "fp16":
        text_encoder_path = convert_fp16_model(params, "text_encoder")
        unet_path = convert_fp16_model(params, "unet")
        vae_decoder_path = convert_fp16_model(params, "vae_decoder")
    else:
        text_encoder_path = os.path.join(params["ori_path"], params["text_encoder"])
        unet_path = os.path.join(params["ori_path"], params["unet"])
        vae_decoder_path = os.path.join(params["ori_path"], params["vae_decoder"])
    
    text_encoder_sess = ort.InferenceSession(text_encoder_path, providers=providers)
    unet_sess = ort.InferenceSession(unet_path, providers=providers)
    vae_decoder_sess = ort.InferenceSession(vae_decoder_path, providers=providers)
    

    thr_num = int(th_num)
    sd_text2img_models = []
    for _ in range(thr_num):
        sd_text2img_models.append(Text2ImgModelSess(params, text_encoder_sess, unet_sess, vae_decoder_sess))
    
    tsv_file_path = './data/PartiPrompts.tsv'
    dataset = []
    cnt = 0
    with open(tsv_file_path, 'r', encoding='utf-8') as tsvfile:
        tsvreader = csv.DictReader(tsvfile, delimiter='\t')
        for row in tsvreader:
            cnt += 1
            row["image_name"] = [f"{cnt:04d}.png"]
            row["category"] = row.pop("Category")
            row["prompt"] = row.pop("Prompt")
            row["image"] = None
            dataset.append(row)
    
    
    print("warmup")
    for data in dataset[:1]:
        # warmup
        _ = sd_text2img_models[0](data["prompt"])
        
    results = []
    print("Start Infer")
    start = time.time()
    results = check_accuracy_multithread(sd_text2img_models, int(batchsize), dataset[:], thr_num)
    end = time.time()
    print(f"Cost time: {end-start}")
    fps = len(dataset[:]) / (end-start)

    print("Infer Finished...")

    print("Start Save images....")
    if False:
        if not os.path.exists("./img/"):
                os.makedirs("./img/")
        for result in results:
            result["image"].save(f"./img/{result['image_name'][0]}")
    

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
    print("StableDiffusion_{}_bs{}_prec{} FPS : {:.3f}, latency : {:.3f}ms".format(model_type, batchsize, "fp16", fps, 1.0/fps*1000))
    logging.info("StableDiffusion_{}_bs{}_prec{} FPS : {:.3f}, latency : {:.3f}ms".format(model_type, batchsize, "fp16", fps, 1.0/fps*1000))
    
    print("StableDiffusion_{}_bs{}_prec{} Avg Score : {:.3f}".format(model_type, batchsize, "fp16", average_score))
    logging.info("StableDiffusion_{}_bs{}_prec{} Avg Score : {:.3f}".format(model_type, batchsize, "fp16", average_score))



if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]

    task = sys.argv[4] 
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    th_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname,batchsize,precision,task,model_path, EP, th_num)
