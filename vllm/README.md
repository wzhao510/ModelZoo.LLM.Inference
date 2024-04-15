## 目录结构及说明

```
.
├── code
│   ├── openai_chatcompletion_client.py
│   ├── openai_completion_client.py
│   ├── run_benchmark_latency.sh
│   ├── run_benchmark_serving.sh
│   ├── run_benchmark_throughput.sh
│   ├── run_offline_inference_demo.sh
│   └── src
│       ├── benchmark_latency.py
│       ├── benchmark_serving.py
│       ├── benchmark_throughput.py
│       ├── offline_inference.py
│       └── template_chatml.jinja
├── dataset
│   └── ShareGPT_V3_unfiltered_cleaned_split.json
├── docker
│   ├── docker-build.sh
│   └── Dockerfile
└── README.md
```

## 1、本地推理脚本run_offline_inference_demo.sh
    脚本内需要根据使用模型情况修改模型所在目录，如/external/models/llama-2-7b-hf
    可根据需要修改code/src/offline_inference.py内的prompts

## 2、benchmark_latency
##    Benchmark the latency of processing a single batch of requests.
    脚本内需要根据使用模型情况修改模型所在目录，如/external/models/llama-2-7b-hf
    可根据测试需要在脚本内增加input-len、output-len、batch-size等参数，详细参数信息见code/src/benchmark_latency.py

## 3、benchmark_throughput
##    Benchmark offline inference throughput.
    脚本内需要根据使用模型情况修改模型所在目录，如/external/models/llama-2-7b-hf
    可根据测试需要在脚本内修改input-len、output-len、batch-size等参数，详细参数信息见code/src/benchmark_throughput.py

## 4、benchmark_serving
##    Benchmark online serving throughput.
    执行benchmark前需要有对应服务启动，简易启动命令: python -m vllm.entrypoints.api_server --model /external/models/llama-2-7b-hf
    服务启动后可在脚本内修改参数，详细参数信息见code/src/benchmark_serving.py

## 5、启动openai_api服务端 
    简易启动命令: python -m vllm.entrypoints.openai.api_server --model /external/models/llama-2-7b-hf --host localhost --port 8000 --chat-template /workspace/ModelZoo.LLM.Inference/vllm/code/src/template_chatml.jinja
    /workspace/ModelZoo.LLM.Inference/vllm/code目录下包含completion和chatcompletion两个客户端sample
    


