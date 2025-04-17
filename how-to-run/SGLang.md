这一页说明如何使用SGLnag框架推理LLM

## 目录结构及说明
```
sgl
├── code
│   ├── sglang
│   │   ├── bench_offline_throughput.py
│   ├── utils
│   ├── bench_test.py
│   ├── run_bench_test_batched.sh
│   ├── run_bench_test_once.sh
│   ├── openai_chatcompletion_client.py
│   ├── openai_completion_client.py
│   ├── run_ceval_client.py
│   ├── run_ceval_test.sh
├── data
│   └── ShareGPT_V3_unfiltered_cleaned_split.json(需要自己准备，具体下载地址见下文)
├── models
└── README.md
```
code下存放的是测试代码和脚本,data目录下是benchmark测试需要用到的数据集，models下存放的是支持的模型的配置文件

## benchmark throughput 执行
* 需要准备ShareGPT_V3_unfiltered_cleaned_split.json数据，请从https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json下载
* 跑看护的35个case性能数据:
```
python ./code/bench_test.py \
--model ./models/DeepSeek-R1-Distill-Qwen-7B/ \
--dataset-path /pde_ai/datasets/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json \
--enable-ep-moe \
--ep-size 1 \
--enable-dp-attention \
--dp-size 1 \
--random-input-len 64 \
--random-output-len 32 \
--num-prompts 1 \
--batched-test 
```
或者在code目录运行run_bench_test_batched.sh
* 跑单次throughput:
```
python ./code/bench_test.py \
--model ./models/DeepSeek-R1-Distill-Qwen-7B/ \
--dataset-path /pde_ai/datasets/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json \
--enable-ep-moe \
--ep-size 1 \
--enable-dp-attention \
--dp-size 1 \
--random-input-len 64 \
--random-output-len 32 \
--num-prompts 1 \

参数说明：
--enable-ep-moe --ep-size 1 开启ep-moe,设置ep-size大小
--enable-dp-attention --dp-size 1 开始dp-attention，设置dp-size大小
--random-input-len  输入长度
--random-output-len 输出长度
--num-prompts 测试batch-size

```
或者在code目录运行run_bench_test_once.sh
* 脚本内需要根据使用模型情况修改模型所在目录
* 可根据测试需要修改并行方式input-len、output-len、batch-size等参数


## ceval精度测试 执行
需要安装open ai server

```
pip install eval-type-backport
```

* 首先需要启动sglang server端：
```
python3 -m sglang.launch_server --model /pde_ai/models/llm/DeepSeek/DeepSeek-R1-Distill-Qwen-7B --trust-remote-code --disable-cuda-graph
```
* 然后运行client端：
```
python ./run_ceval_client.py --model /pde_ai/models/llm/DeepSeek/DeepSeek-R1-Distill-Qwen-7B --test_jsonl /pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl
```
* 测试集ceval_val_cmcc.jsonl需要自行准备