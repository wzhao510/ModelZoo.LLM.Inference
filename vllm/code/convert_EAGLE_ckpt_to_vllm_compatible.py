import json

import torch
from safetensors.torch import load_file, save_file
import sys
import os

ckptpath= sys.argv[1] #/pde_ai/models/llm/Llama//EAGLE-LLaMA3-Instruct-8B/pytorch_model.bin
ref_ckptpath=sys.argv[2] #/pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct/model-00004-of-00004.safetensors

ckpt = torch.load(ckptpath)
ref_ckpt = load_file(ref_ckptpath)
#print(ref_ckpt.keys())

ckpt['lm_head.weight'] = ref_ckpt['lm_head.weight']

parent_folder_path = os.path.dirname(ckptpath)
#print(f"path {parent_folder_path}")
newfilepath =f"{parent_folder_path}_vllm"
if not os.path.exists(newfilepath):
    os.mkdir(newfilepath) 

save_file(ckpt, f"{newfilepath}/model.safetensors")

with open(f"{parent_folder_path}/config.json") as rf:
    cfg = json.load(rf)

if "fc.bias" in ckpt:
    cfg["eagle_fc_bias"] = True

cfg = {"model_type": "eagle", "model": cfg}

with open(f"{newfilepath}/config.json", "w") as wf:
    json.dump(cfg, wf)