import json

Jsonfile = "./%s/config.json"

def get_params(modelname):
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
    return param_all


def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)
    