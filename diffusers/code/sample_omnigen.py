import sys
import argparse
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from OmniGen import OmniGenPipeline
#from utils.utils import get_params
import csv
import time
from datetime import datetime
import os

pipe =None


def loadmodel(modelpath,offload=False):
    global pipe
    pipe = OmniGenPipeline.from_pretrained(modelpath)
    if offload:
        print("cpu offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

def T2I(bs=1,w=1024,h=1024,gs=3,steps=50,issaveimg=False,outputpath="output/omnigen"):
    global pipe
    prompt =["Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, \
        with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile.\
        Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD.",
        "A curly-haired man in a red shirt is drinking tea.",
        "a tiny astronaut hatching from an egg on the moon",
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
        "a corgi’s head depicted as an explosion of a nebula"
        ]
    start_time = time.time()
    images = pipe(
        prompt=prompt[:bs],
        height=w,
        width=h,
        guidance_scale=gs,
        num_inference_steps=steps,
        seed=0
    )
    end_time = time.time()
    elapsed_time = (end_time-start_time)*1000
    data=[prompt[0],bs,f"{w}x{h}",round(elapsed_time,2)]
    if issaveimg:
        for i in range(len(images)):
            images[i].save(f"{outputpath}/bs{args.batchsize}_image_{i}_{w}x{h}.png")
    return data

def I2I(imgs:Union[List[str], List[List[str]]] = None,w=1024,h=1024,gs=3,igs=1.6,steps=50,issaveimg=False,outputpath="output/omnigen"):
    global pipe
    prompt="A professor and a boy are reading a book together. The professor is the middle man in <img><|image_1|></img>. The boy is the boy holding a book in <img><|image_2|></img>."
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        input_images=imgs,
        height=h,
        width=w,
        guidance_scale=gs,
        img_guidance_scale=igs,
        separate_cfg_infer=True,
        seed=0)
    end_time = time.time()
    elapsed_time = (end_time-start_time)*1000
    data=[prompt,1,f"{w}x{h}",round(elapsed_time,2)]
    if issaveimg:
        images[0].save(f"{outputpath}/I2I_image_{w}x{h}.png") 
    return data  
def main(args):
     model =  os.path.basename(args.model.rstrip("/"))
     steps =50
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run flux demo")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="model path")
    parser.add_argument("--batchsize",
                        type=int,
                        default=1,
                        help="Number of prompts for throughput test")
    parser.add_argument("--save_image",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="save image or not")
    parser.add_argument("--offload", type=lambda x: (str(x).lower() == 'true'),  # 将字符串转换为布尔值
                        default=False,  # 默认值为 True
                        help="offload true or false")
    parser.add_argument("--resolution",
                        type=str,
                        default="1024x1024",
                        help="The resolution of the target image")
    parser.add_argument("--infertype",
                        type=str,
                        required=True,
                        help="T2I OR I2I, if I2I the batchsize must 1")
    args = parser.parse_args()
    loadmodel(args.model,args.offload)
    outputpath = "output/omnigen"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    
    img_info_s = args.resolution.split("x")

    output_height = int(img_info_s[1])
    output_width = int(img_info_s[0])

    csv_data=[]
    headers = ["promt", "bs", "hw", "e2e time"]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfilename=f"output/omnigen/{args.infertype}_{args.resolution}_bs{args.batchsize}_output_{current_time}.csv"
    csv_data.append(headers)
    if args.infertype == "T2I":
        dwu = T2I(bs=args.batchsize,w=output_width,h=output_height,
            issaveimg = args.save_image,
            outputpath= outputpath)
        # with torch.profiler.profile(
        #             activities=[
        #                 torch.profiler.ProfilerActivity.CUDA,
        #                 torch.profiler.ProfilerActivity.CPU,
        #             ],
        # ) as prof:
        u_data = T2I(bs=args.batchsize,w=output_width,h=output_height,
                issaveimg = args.save_image,
                outputpath= outputpath)
        csv_data.append(u_data)
        # prof.export_chrome_trace("T2Itrace.json")
        # pertable=prof.key_averages().table(sort_by="cuda_time_total",row_limit=50)
        # print('====================profile===================')
        # print(pertable)
        # print('====================profilr===================')
        
        # with open("T2I_profile_table.txt", "w") as f:
        #     f.write(pertable)
    elif args.infertype == "I2I":
        input_images=["./data/AI_Pioneers.jpg", "./data/same_pose.png"]
        dwu = I2I(imgs=input_images,w=output_width,h=output_height,
            issaveimg = args.save_image,
            outputpath= outputpath)
        # with torch.profiler.profile(
        #             activities=[
        #                 torch.profiler.ProfilerActivity.CUDA,
        #                 torch.profiler.ProfilerActivity.CPU,
        #             ],
        # ) as prof:
        u_data = I2I(imgs=input_images,w=output_width,h=output_height,
            issaveimg = args.save_image,
            outputpath= outputpath)
        csv_data.append(u_data)
        # prof.export_chrome_trace("I2Itrace.json")
        # pertable=prof.key_averages().table(sort_by="cuda_time_total",row_limit=50)
        # print('====================profile===================')
        # print(pertable)
        # print('====================profilr===================')
        
        # with open("I2Iprofile_table.txt", "w") as f:
        #     f.write(pertable)
    
    with open(outfilename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in csv_data:
            writer.writerow(row)