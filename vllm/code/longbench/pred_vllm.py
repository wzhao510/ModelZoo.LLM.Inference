# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, )
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--s', type=int, help='model size in B')
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--checkpoint', type=str, help="checkpoint_path")
    parser.add_argument('--quantize',  action='store_true', help="Debug mode")
    parser.add_argument('--tp', type=int, help='tp')

    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
   


    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    tp = args.tp
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["trec",
 "samsum","triviaqa"]
                  #  
    else:
        #datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #           "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = ["multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh", "lcc"]
        #datasets = ["lcc"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    print(args.checkpoint)
    #import pdb       
    #pdb.set_trace()
    if args.quantize:
            model = LLM(model=args.checkpoint,tensor_parallel_size=tp,trust_remote_code=True,quantization="GPTQ",max_model_len=15000)
    else:
        #import pdb
        #pdb.set_trace()
        model = LLM(model=args.checkpoint,tensor_parallel_size=tp,trust_remote_code=True, gpu_memory_utilization=0.9)
 
    for dataset in datasets:
        if args.e:
            data=load_dataset("json", data_files=f"/pde_ai/datasets/projects/longbench_data/data/{dataset}_e.jsonl", split="train")
            # data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("json", data_files=f"/pde_ai/datasets/projects/longbench_data/data/{dataset}.jsonl", split="train")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        if os.path.exists(out_path):
            os.remove(out_path)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if int(args.s)>18:
            world_size=8
        else:
            world_size = 8
        print(world_size)
        if args.debug:
            world_size=1
        data_split=8//world_size
        data_subsets = [data_all[i::data_split] for i in range(data_split)]
        sampling_params = SamplingParams(max_tokens=dataset2maxlen[dataset],use_beam_search=True,n=2,
                        temperature=0.0)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            output=model.generate(prompt, sampling_params)
            pred=output[0].outputs[0].text
            if pred=='':
                print(output)
           
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
            # get_pred(0,1,data_all,max_length,max_gen,prompt_format,dataset,device,model_name,model2path,out_path)
        
