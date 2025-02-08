这一页说明如何使用vLLM框架推理LLM。建议使用vllm Docker环境，创建容器时添加模型目录映射：-v /pde_ai:/pde_ai，测试工程需放在/workspace下。

目前vLLM已支持的模型包括：
Aquila2_34b Aquila_7b 
Baichuan2_13b_chat Baichuan2_7b_chat Baichuan_7b 
ChatGLM2_6b ChatGLM3_6b ChatGLM3_6b_32k 
CodeGeeX2_6b 
CodeLlama_13b CodeLlama_7b 
DeciLM-7B 
InternLM_7b 
LLama3.1_70b LLama3.1_8b LLama3_70b LLama3_8b 
Llama2_13b Llama2_70b_chat Llama2_7b Llama2_7b_int4_awq Llama2_7b_int4_gptq Llama2_7b_sql_lora Llama_13b Llama_30b Llama_65b Llama_7b 
Mistral-7B-v0.1 
Mixtral-8x7B-V0.1 
Qwen2_72b Qwen2_7b Qwen_14b_chat Qwen_1_5_14b Qwen_1_5_14b_int8_gptq Qwen_1_5_32b Qwen_1_5_72b Qwen_1_5_7b Qwen_72b_chat Qwen_7b 
Yi_34b Yi_6b 
bloom-7b1 
folcon_40b folcon_7b 
glm4_9b_chat 
gpt4all-j gpt_bigcode-santacoder 
mpt-30b 
pythia-12b 
vicuna-13b-v1.5


## vllm目录结构及说明
```
.
├── code
│   ├── bench_test.py
│   ├── c-eval.py
│   ├── openai_chatcompletion_client.py
│   ├── openai_completion_client.py
│   ├── run_benchmark_latency.sh
│   ├── run_benchmark_serving.sh
│   ├── run_benchmark_throughput.sh
│   ├── run_offline_inference_demo.sh
│   ├── src
│   ├── tools
│   └── utils
├── dataset
│   └── ShareGPT_V3_unfiltered_cleaned_split.json(需要自己准备，具体下载地址见下文-benchmark_serving)
├── data
│   ├── demo.jpeg
│   └── demo.jpg
├── models
│   ├── Aquila2_34b
│   ├── Aquila_7b
│   ├── Baichuan2_13b_chat
│   ├── Baichuan2_7b_chat
│   ├── Baichuan_7b
│   ├── bloom-7b1
│   .
|   .
|   .


code下存放的是测试代码和脚本，dataset是启动openai_api服务端用到的，models下存放的是支持的模型的配置文件，multimodal_test是多模态模型测试。
```

## C-Eval 精度测试
1. 安装依赖（默认已预装）
```shell
pip install lm-eval==0.4.2
```
安装原生的 0.4.2 之后，需要修改 安装lm-eval路径，比如  
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
cd /workspace/ModelZoo.LLM.Inference/vllm
ln -s /pde_ai/datasets/dataset-7/ModelZoo_LLM_data/ceval ceval
ln -s /pde_ai/datasets/dataset-7/ModelZoo_LLM_data/lm_eval_code/exact_match exact_match
```
3. 执行
```
python code/c-eval.py ./models/${Model_name}
其中${Model_name}为models中目录名。
```

## benchmark throughput 性能测试
使用方式：
```
python code/bench_test.py  ./models/${Model_name} 24                # 跑24条数据测试 ，默认： 输入长度1024 输出长度1024
python code/bench_test.py  ./models/${Model_name} 1024 512 128      # 跑1024条数据测试，设置: 输入长度512 输出长度 128
python code/bench_test.py  ./models/${Model_name} 1024 512 128 1     # 最后一个1 表示进行批次跑；前面设置批次、输入长度、输出长度不生效，将会一次加载模型跑看护的35个case性能数据
其中${Model_name}为models中目录名。
```
----  2024.07.31  ----
### 新增环境变量 `MX_VLLM_ENABLE_PROFILE`
* 使能 `MX_VLLM_ENABLE_PROFILE`环境变量后将会在 `./mx_profiler/` 文件夹通过torch_profiler 工具生成csv原始文件（如果跑35个case的话，目前只统计 input_len=256,output_len=128 以及 input_len=1024,output_len=1024 数据）

生成的对应文件夹路径下的csv 可以通过 以下脚本完成 kernel 汇总（注：需要 安装openxl包： `pip install openxl`）
```shell
python ./vllm/code/tools/statistics_csv.py ./mx_profiler/
```

## LoRA 特性支持 （Released版本大于等于 2.23）benchmark 
离线推理脚本（配置 lora_path 为微调的LoRA模型路径）：
```
python code/src/offline_inference_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /pde_ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```
multi-LoRA 推理脚本：
```python
python code/src/offline_inference_multi_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /pde_ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```
LoRA 跑性能数据 --当前性能较差，后续会对其进行优化
```python
python code/bench_test.py ./models/Llama2_7b_sql_lora/ 64 1024 1024
```

## 量化特性支持（Released版本大于等于 2.23）
用法同普通一致，模型路径要为gptq/awq 模型路径，
```python
python code/src/offline_inference.py --model /pde_ai/models/llm/quantize_model/llama-2-7b-int4-gptq/
```

## 多模态模型 特性支持（Released版本大于等于 2.25.2）
简易启动命令：
```
 python code/run_multimodal.py --model ./models/InternVL-chat-v1.5/
```


## 1、本地推理demo
脚本为 code/run_offline_inference_demo.sh：
```
python src/offline_inference.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf
```
运行该脚本：
```shell
bash run_offline_inference_demo.sh
```
脚本内需要根据使用模型情况修改模型所在目录，如 --model /pde_ai/models/llm/Llama/Llama-2-7b-hf;  
可根据需要修改code/src/offline_inference.py内的prompts;  
详细参数信息见code/src/offline_inference.py

## 2、benchmark_latency
Benchmark the latency of processing a single batch of requests.  脚本为 code/run_benchmark_latency.sh: 
```
python src/benchmark_latency.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf
```
运行该脚本：
```shell
bash run_benchmark_latency.sh
```
脚本内需要根据使用模型情况修改模型所在目录，如 --model /pde_ai/models/llm/Llama/Llama-2-7b-hf;   
可根据测试需要在脚本内增加input-len、output-len、batch-size等参数;  
详细参数信息见code/src/benchmark_latency.py

## 3、benchmark_throughput
Benchmark offline inference throughput. 脚本为 /code/run_benchmark_throughput.sh:
```
python src/benchmark_throughput.py --backend hf --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --tokenizer /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --input-len 512 --output-len 128 --hf-max-batch-size 8
```
运行该脚本：
```shell
bash run_benchmark_throughput.sh
```
脚本内需要根据使用模型情况修改模型所在目录，如 --model /pde_ai/models/llm/Llama/Llama-2-7b-hf;   
可根据测试需要在脚本内增加input-len、output-len、batch-size等参数;  
详细参数信息见code/src/benchmark_throughput.py

## 4、benchmark_serving
Benchmark online serving throughput.  
执行online benchmark前需要有对应服务启动，简易启动命令: 
```
python -m vllm.entrypoints.api_server --model /pde_ai/models/llm/Llama/Llama-2-7b-hf
```
脚本为 /code/run_benchmark_serving.sh:
```
python src/benchmark_serving.py --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --num-prompts 10
```
服务启动后可在脚本内修改参数，详细参数信息见code/src/benchmark_serving.py
执行测试脚本：
```
bash run_benchmark_serving.sh
```

## 5、启动openai_api服务端 
```
pip install eval-type-backport
```
简易启动命令: 
```
python -m vllm.entrypoints.openai.api_server --model /pde_ai/models/llm/Llama/Llama-2-7b-hf --host localhost --port 8000 --chat-template /workspace/ModelZoo.LLM.Inference/vllm/code/src/template_chatml.jinja
```
/workspace/ModelZoo.LLM.Inference/vllm/code目录下包含completion和chatcompletion两个客户端sample
```
python openai_chatcompletion_client.py
```
```
python openai_completion_client.py
```

## 6、测试vllm 0.6.6 APC （Automatic Prefix Caching）
参数说明
--model 模型路径

使用pytext 启动 APC 命令
```
python code/test_prefix_caching.py --model /pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct/
```

## 7、测试vllm 0.6.6 Chunked Prefill 
#默认max_num_batched_tokens=512
```
python ./code/bench_test.py --model models/Llama_7b  --enable-chunked-prefill
#注意：--model 指的是 模型名称，如果在镜像中运行请确认模型路径是否正确，模型路径配置在：./models/Llama_7b/config.json 中的 model_path 字段。
```
## 8.Speculative Decoding
参数说明
--model 主模型路径
--prompts 提示词，""The future of AI is","hello"
--tensor-parallel-size 张量并行数量 默认值 1
--speculative-model speculative 模型的路径，MLP Speculators ，EAGLE based draft models 等。
--num-speculative-tokens 推测令牌 数量，3-10 ，默认值 5
--ngram 开启Speculating by matching n-grams in the prompt
--ngram-prompt-lk-max  looup  最大数量，默认值4

注意事项：
HF 上下载的EAGLE 模型文件不能直接被vllm 使用，可以使用convert_EAGLE_ckpt_to_vllm_compatible.py 转换为vllm 可用的模型文件和配置文件。
命令如：
```
python code/convert_EAGLE_ckpt_to_vllm_compatible.py /pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin /pde_ai/models//llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors
```
参数及结果说明：
/pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin 为原始模型权重
/pde_ai/models//llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors 主模型的权重最后一个分片
转换后的文件保存在： /pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct_vllm/ 规则  EAGLE-xxx_vllm

```
python code/test_Speculative_Decoding.py --model $modelpath --prompts "The future of AI is" --speculative-model $spmodel
```
#MLP 推测解码
```
python code/test_Speculative_Decoding.py --model /pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct --prompts "The future of AI is" --speculative-model /pde_ai/models/llm/Llama/llama3-8b-accelerator
```
#使用4张GPU卡
#speculative models currently need to be run without tensor parallelism
python code/test_Speculative_Decoding.py --model /pde_ai/models/llm/Llama/Meta-Llama-3.1-70B-Instruct  --tensor-parallel-size 4 --prompts "The future of AI is" --speculative-model /pde_ai/models/llm/Llama/llama3-70b-accelerator

#使用 ngram
python code/test_Speculative_Decoding.py --model /pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct --prompts "The future of AI is" --ngram  --ngram-prompt-lk-max 4

#使用 eagle draft model
python code/test_Speculative_Decoding.py --model /pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct --prompts "The future of AI is" --speculative-model /pde_ai/models/llm/Llama/EAGLE-LLaMA3-Instruct-8B