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

## C-Eval 执行
1. 依赖 环境（默认已预装）
```shell
lm-eval=0.4.2
```

安装原生的 0.4.2  需要修改 安装lm-eval路径，比如  
/opt/conda/lib/python3.8/site-packages/lm_eval/api/model.py ：294  _encode_pair 方法为下面的：  
```python
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc
```
2. 建立软连接：

```shell
ln  -s /AI-DATA/dataset/ModelZoo_LLM_data/ceval ceval
ln  -s /AI-DATA/dataset/ModelZoo_LLM_data/lm_eval_code/exact_match exact_match
```
3. 执行方式
python code/c-eval.py  ./models/xxxx  # 

## benchmark throghtout 执行
python code/bench_test.py  ./models/xxxx 24 # 跑24条数据测试
python code/bench_test.py  ./models/xxxx 1024 # 跑1024条数据测试

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
    


