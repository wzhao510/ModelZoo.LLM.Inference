import sys
import argparse
import torch
from diffusers import FluxPipeline
from utils.utils import get_params


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

def main(args):
    params = get_params(args.model)

    img_info = params["outputs_size"].split("#")
    output_height = int(img_info[0])
    output_width = int(img_info[1])
    num_inference_steps = params["num_inference_steps"]

    pipe = FluxPipeline.from_pretrained(params["ori_path"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    out = pipe(
        prompt=prompt[:args.batchsize],
        guidance_scale=3.5,
        height=output_height,
        width=output_width,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(params["seed"])
    ).images

    if args.save_image:
        for i in range(len(out)):
            out[i].save(f"image_{i}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run flux demo")
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--batchsize",
                        type=int,
                        default=32,
                        help="Number of prompts for throughput test")
    parser.add_argument("--save-image",
                        type=str2bool,
                        default=True,
                        help="save image or not")
    args = parser.parse_args()
    main(args)