import sys
import argparse
import torch
from diffusers import FluxPipeline
from utils.utils import get_params
import csv
import time
from datetime import datetime
import os

prompt = [
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
    "a corgi’s head depicted as an explosion of a nebula",
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def warmupflux(pipe,gs,width,height,steps=2):
    out = pipe(
            prompt=prompt[0],
            guidance_scale=gs,
            height=int(height),
            width=int(width),
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images
    
def main(args):
    headers = ["promt", "bs", "hw", "e2e time"]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists("output/"):
        os.makedirs("output/")


    csv_data=[]
    model =  os.path.basename(args.model.rstrip("/"))
    params = get_params(f"models/{model}")
    modelpath = args.model
    if not os.path.isdir(modelpath) :
        modelpath= params["ori_path"]

    outfilename=f"output/{model}_{args.resolution}_bs{args.batchsize}_output_{current_time}.csv"
    csv_data.append(headers)
    #读取配置
    num_inference_steps = params["num_inference_steps"]
    guidance_s=params['guidance_scale']
    pipe = FluxPipeline.from_pretrained(modelpath, torch_dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    img_info_s = args.resolution.split("x")
    #wram up
    warmupflux(pipe,guidance_s,img_info_s[0],img_info_s[1])
    if args.batchsize>1 :  
        output_height = int(img_info_s[1])
        output_width = int(img_info_s[0])

        start_time = time.time()
        out = pipe(
            prompt=prompt[:args.batchsize],
            guidance_scale=guidance_s,
            height=output_height,
            width=output_width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(params["seed"])
        ).images
        end_time = time.time()
        elapsed_time = (end_time-start_time)*1000
        if args.save_image:
            for i in range(len(out)):
                out[i].save(f"output/bs{args.batchsize}_image_{i}_{output_height}x{output_width}.png")
        data=[prompt[0],args.batchsize,f"{output_height}x{output_width}",round(elapsed_time,2)]
        csv_data.append(data)

        with open(outfilename, mode="w", newline="") as file:
            writer = csv.writer(file)
            for row in csv_data:
                writer.writerow(row)
    else:
        #第一次的 warmup 性能是有些慢的，bs=1 的时候我们跑2组数据出来。
        output_height = int(img_info_s[0])
        output_width = int(img_info_s[1])
        start_time = time.time() 
        out = pipe(
            prompt=prompt[0],
            guidance_scale=guidance_s,
            height=output_height,
            width=output_width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(params["seed"])
        ).images
        end_time = time.time()
        elapsed_time = (end_time-start_time)*1000
        data=[prompt[0],1,f"{output_height}x{output_width}",round(elapsed_time,2)]
        csv_data.append(data)
        if args.save_image:
            out[0].save(f"output/bs1_image_{output_height}x{output_width}.png")

        with open(outfilename, mode="w", newline="") as file:
            writer = csv.writer(file)
            for row in csv_data:
                writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run flux demo")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Number of prompts for throughput test")
    parser.add_argument("--batchsize",
                        type=int,
                        default=1,
                        help="Number of prompts for throughput test")
    parser.add_argument("--save-image",
                        type=str2bool,
                        default=True,
                        help="save image or not")
    parser.add_argument("--offload", type=lambda x: (str(x).lower() == 'true'),  # 将字符串转换为布尔值
                        default=False,  # 默认值为 True
                        help="offload true or false")
    parser.add_argument("--resolution",
                        type=str,
                        default="1024x1024",
                        help="Number of prompts for throughput test")
    args = parser.parse_args()
    main(args)