import json
import subprocess

Jsonfile = "./%s/config.json"

def get_params(modelname):
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
    return param_all


def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

def get_vllm_version():
    result = subprocess.run(['pip', 'show', 'vllm'], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            return line.split()[-1]
    return None
    