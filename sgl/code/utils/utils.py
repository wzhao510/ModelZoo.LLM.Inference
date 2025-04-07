# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import json

from pathlib import Path

_model_dir = Path(__file__).parents[2].resolve(strict=True)


def get_modelzoo_vllm_dir():
    return _model_dir


def get_model_config_filename(modelname):
    return _model_dir / modelname / "config.json"


def get_params(modelname):
    with open(get_model_config_filename(modelname), "r") as f:
        param_all = json.load(f)
    return param_all


def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

