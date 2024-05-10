import fire
from macaPMX.model_zoo.llama3.huggingface import run_export

if __name__ == '__main__':
    fire.Fire(run_export)