# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import json
import cpuinfo 

from pathlib import Path

_model_dir = Path(__file__).parents[2].resolve(strict=True)


def get_modelzoo_vllm_dir():
    return _model_dir


def get_model_config_filename(modelname):
    return _model_dir / modelname / "config.json"


def get_model_common_filename(file):
    return f"{_model_dir}/models/common/{file}"


def get_params(modelname):
    with open(get_model_config_filename(modelname), "r") as f:
        param_all = json.load(f)
    return param_all

def get_common_params(file):
    with open(get_model_common_filename(file), "r") as f:
        param_all = json.load(f)
    return param_all


def get_json_string(obj):
    return json.dumps(obj)

def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)


def get_vllm_version():
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        print("Not install vLLM")
        return None
    except AttributeError:
        print("Can not get vLLM version")
        return None

# 获取cpu信息
def get_cpu_info():
    return cpuinfo.get_cpu_info()

# 获取cpu厂商
# intel: GenuineIntel
# AMD: AuthenticAMD
# Hygon(海光): HygonGenuine or  HygonAuthentic
def get_cpu_vendor_id():
    info = cpuinfo.get_cpu_info()
    return info["vendor_id_raw"]