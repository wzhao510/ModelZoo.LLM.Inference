from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_6e39a4 import aime2024_datasets

datasets = aime2024_datasets

from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        abbr='deepseek-r1',
        type=OpenAISDK,
        path='/mnt/dataset/models/llm/DeepSeek/DeepSeek-R1-BF16/',
        openai_api_base='http://localhost:8000/v1',
        key='EMPTY',
        rpm_verbose=True,
        max_out_len=20480,
        max_seq_len=20480, # 32768
        temperature=0.6,
        batch_size=64,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        ),
]
