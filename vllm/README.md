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
|   ├── ....
│   └── src
│       ├── benchmark_latency.py
│       ├── benchmark_serving.py
│       ├── benchmark_throughput.py
│       ├── offline_inference.py
│       ├── template_chatml.jinja
│       └── ....

├── dataset
│   └── ShareGPT_V3_unfiltered_cleaned_split.json(需要自己准备，具体下载地址见下文-benchmark_serving)
├── data
│   ├── demo.jpeg
│   └── demo.jpg
└── README.md
```

## C-Eval 执行
1. 依赖 环境（默认已预装）
```shell
lm-eval=0.4.2
```

安装原生的 0.4.2  需要修改 安装lm-eval路径，比如  
/opt/conda/lib/python3.8/site-packages/lm_eval/api/model.py _encode_pair 方法为下面的：  
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
如果是本地运行需要修改下 路径，否则会一直尝试网络下载。
3. 执行方式
python code/c-eval.py  --model ./models/xxxx --enforce_eager
当前支持参数列表如下：
 - model:&nbsp;&nbsp;原始模型路径，必须设置
 - enforce-eager:&nbsp;&nbsp;是否使用eager模式，默认不配置，若不配置使用cuda graph。

## benchmark throughput 执行
----  2024.11.10  ---
```
python code/bench_test.py  --model ./models/xxxx --num-prompts 24                # 跑24条数据测试 ，默认： 输入长度1024 输出长度1024
python code/bench_test.py  --model ./models/xxxx --num-prompts 32 --input-len 512 --output-len 128      # 跑32条数据测试，设置: 输入长度512 输出长度 128
python code/bench_test.py  --model ./models/xxxx --num-prompts 32 --input-len 512 --output-len 128 --batched-test   # 最后 "--batched-test" 表示进行批次跑；前面设置批次、输入长度、输出长度不生效，将会一次加载模型跑看护的35个case性能数据.
```
当前支持参数列表如下：
 - model:&nbsp;&nbsp;原始模型路径，必须设置
 - num-prompts:&nbsp;&nbsp;测试prompts数量，可以类比为batchsize， 默认为32
 - input-len:&nbsp;&nbsp;测试输入长度， 默认为1024
 - output-len:&nbsp;&nbsp;测试输出长度， 默认为1024
 - batched-test:&nbsp;&nbsp;是否进行批量测试（默认不设置），若设置该配置前面设置批次、输入长度、输出长度不生效，将会一次加载模型跑看护的35个case性能数据.
 - enforce-eager:&nbsp;&nbsp;是否使用eager模式，默认不配置，若不配置使用cuda graph。
 - num-scheduler-steps:&nbsp;&nbsp;每个scheduler对应最大前项步数，默认为1, vllm 0.6.2 开始支持。

----  2024.07.31  ---
### 新增环境变量 `MX_VLLM_ENABLE_PROFILE`
* 使能 `MX_VLLM_ENABLE_PROFILE`环境变量后将会在 `./mx_profiler/` 文件夹通过torch_profiler 工具生成csv原始文件（如果跑35个case的话，目前只统计 input_len=256,output_len=128 以及 input_len=1024,output_len=1024 数据）

生成的对应文件夹路径下的csv 可以通过 以下脚本完成 kernel 汇总（注：需要 安装openxl包： `pip install openpyxl`）
```shell
python ./code/tools/statistics_csv.py ./mx_profiler/
```
## LoRA 特性支持 （Released版本大于等于 2.23）benchmark 
离线推理脚本（配置 lora_path 为微调的LoRA模型路径）：
```python
python code/src/offline_inference_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /AI-DATA/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```
multi-LoRA 推理脚本
```python
python code/src/offline_inference_multi_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /AI-DATA/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```

LoRA 跑性能数据 --当前性能较差，后续会对其进行优化
复用benchmark throughput测试方法，设置lora模型路径即可
```python
python code/bench_test.py --model ./models/Llama2_7b_sql_lora/ --num-prompts 32 --input-len 512 --output-len 128
```

## GPTQ 特性支持（Released版本大于等于 2.23）
用法同普通一致，模型路径要为gptq 模型路径
```python
python code/src/offline_inference.py --model /pde_ai/models/llm/quantize_model/llama-2-7b-int4-gptq/
```

GPTQ 跑性能数据
复用benchmark throughput测试方法，设置gptq模型路径即可
```python
python code/bench_test.py --model ./models/Llama2_7b_int4_gptq/ --num-prompts 32 --input-len 512 --output-len 128
```
## 1、本地推理脚本run_offline_inference_demo.sh
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/Llama/Llama-2-7b-hf/
    可根据需要修改code/src/offline_inference.py内的prompts

## 2、benchmark_latency
##    Benchmark the latency of processing a single batch of requests.
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/Llama/Llama-2-7b-hf/
    可根据测试需要在脚本内增加input-len、output-len、batch-size等参数，详细参数信息见code/src/benchmark_latency.py

## 3、benchmark_throughput
##    Benchmark offline inference throughput.
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/Llama/Llama-2-7b-hf/
    可根据测试需要在脚本内修改input-len、output-len、batch-size等参数，详细参数信息见code/src/benchmark_throughput.py

## 4、benchmark_serving
##    Benchmark online serving throughput.
执行benchmark前需要有对应服务启动，简易启动命令: 
```
vllm serve /pde_ai/models/llm/Llama/Llama-2-7b-hf/ -pp 1 -tp 1  --trust-remote-code --distributed-executor-backend ray --max-model-len 4096 --swap-space 16 --gpu-memory-utilization 0.95
```

服务启动后可在脚本内修改参数，详细参数信息见code/src/benchmark_serving.py
    
若使用code/run_benchmark_serving.sh测试，需要准备ShareGPT_V3_unfiltered_cleaned_split.json数据，请从https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json下载，并拷贝至./dataset路径。

## 5、启动openai_api服务端 
简易启动命令: 
```
vllm serve /pde_ai/models/llm/Llama/Llama-2-7b-hf/ -pp 1 -tp 1  --trust-remote-code --distributed-executor-backend ray --max-model-len 4096 --swap-space 16 --gpu-memory-utilization 0.95
```
    
## 多模态模型 特性支持（Released版本大于等于 2.25.2）
简易启动命令：
```
 python ./code/run_multimodal.py --model ./models/InternVL-chat-v1.5/
```
---- 2025.02.27 ----

新增longbench v1长文本精度测试集:

```
pip install -r code/longbench/requirements.txt 
```

跑GPTQ模型

```
python code/longbench/pred_vllm.py --model qwen2.5-72b-gptq-int4 --checkpoint /pde_ai/models/llm/Qwen/Qwen2___5-72B-Instruct-GPTQ-Int4/ --s 8 --tp 2 --quantize
```

跑非GPTQ模型
'''
python pred_vllm.py --model qwen2-7b --checkpoint /pde_ai/models/llm/Qwen2-7B-Instruct --tp 1  --s 8
'''
通过开源脚本https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py得到模型的longbench结果,保存在./pred
'''
python eval.py --model qwen2-7b
'''

---- 2025.03.12 ----

新增client ceval测试方法：
vllm server 
```
vllm serve ${model_path}  --trust-remote-code
```

数据集默认路径 /pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl

```
python ./code/run_ceval_client.py --model ${model_path, 路径最后加“/”} -b ${请求并发数，默认为8} --save_dir ${结果路径，默认为 ./results}
```
