from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

datasets = mmlu_datasets


models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-pytorch',
        path="/external/ai/models/llm/Internlm/internlm2-chat-7b",
        backend="pytorch",
        engine_config=dict(device_type="maca", block_size=256, dtype="float16", tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
