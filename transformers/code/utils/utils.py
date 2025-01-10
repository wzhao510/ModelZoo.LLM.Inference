import json
from datetime import datetime
from pathlib import Path
import os
import warnings
_model_dir = Path(__file__).parents[2].resolve(strict=True)

def get_model_performeance_filename(modelname):
    return _model_dir / modelname / "performance.json"

def get_model_config_filename(modelname):
    return _model_dir / modelname / "config.json"


def get_params(modelname):
    with open(get_model_config_filename(modelname), 'r') as f:
        param_all = json.load(f)
    return param_all

def set_gpu(tp):
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not cuda_devices:
        if tp == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif tp == 4:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        elif tp == 8:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        else:
            warnings.warn("For unconventional TP numbers (1, 4, 8), it is recommended to manually configure \"CUDA_VISIBLE_DEVICES\" to specify which GPUs to use.")

def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

def update_json_file(json_path, test_dataset, data_type, current_acc):
    """
    更新JSON文件,如果文件不存在则创建新的JSON结构并写入对应数据
    如果文件存在，则对比当前准确率和已有的最佳准确率，按规则进行替换更新
    """
    current_date = datetime.now().strftime('%Y%m%d')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if test_dataset in data:
            if data_type in data[test_dataset]:
                best_acc = data[test_dataset][data_type]['Best Acc']
                best_date = data[test_dataset][data_type]['Best Acc Date']
                if current_acc > best_acc:
                    data[test_dataset][data_type]['Best Acc'] = current_acc
                    data[test_dataset][data_type]['Best Acc Date'] = current_date
                data[test_dataset][data_type]['Current Acc'] = current_acc
                data[test_dataset][data_type]['Date'] = current_date
            else:
                data[test_dataset][data_type] = {
                    'Best Acc': current_acc,
                    'Best Acc Date': current_date,
                    'Current Acc': current_acc,
                    'Date': current_date
                }
        else:
            data[test_dataset] = {
                data_type: {
                    'Best Acc': current_acc,
                    'Best Acc Date': current_date,
                    'Current Acc': current_acc,
                    'Date': current_date
                }
            }
    else:
        data = {
            test_dataset: {
                data_type: {
                    'Best Acc': current_acc,
                    'Best Acc Date': current_date,
                    'Current Acc': current_acc,
                    'Date': current_date
                }
            }
        }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, use_logit_bias=False, only_load=None, device_map="auto", device="cuda"):
#     kwargs = {"device_map": device_map}

#     if device != "cuda":
#         kwargs['device_map'] = {"": device}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if model_base is not None:
#         # PEFT model
#         from peft import PeftModel, PeftConfig
#         tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=False)
#         model = AutoModelForCausalLM.from_pretrained(model_base, trust_remote_code=True, low_cpu_mem_usage=True, **kwargs)
#         print(f"Loading LoRA weights from {model_path}")
#         lora_config = PeftConfig.from_pretrained(model_path)
#         if only_load == "attn":
#             lora_config.target_modules = {m for m in lora_config.target_modules if m not in ["up_proj", "down_proj", "gate_proj"]}

#         elif only_load == "ffn":
#             lora_config.target_modules = {m for m in lora_config.target_modules if m in ["up_proj", "down_proj", "gate_proj"]}
        
#         model = PeftModel.from_pretrained(model, model_path, config=lora_config)
#         print(f"Merging weights")
#         model = model.merge_and_unload()
#         print('Convert to FP16...')
#         model.to(torch.float16)

#         tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base, add_prefix_space=True, trust_remote_code=True)
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

#         tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048
        
#     return tokenizer, model, context_len, tokenizer_with_prefix_space