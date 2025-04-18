# VLLM 项目教程及说明

***
> 这一页说明如何使用vLLM框架推理LLM。建议使用vllm Docker环境，创建容器时添加模型目录映射：-v /pde_ai:/pde_ai，测试工程需放在/workspace下。
> 
> 目前vLLM已支持的模型包括：
> Aquila2_34b Aquila_7b 
> Baichuan2_13b_chat Baichuan2_7b_chat Baichuan_7b 
> ChatGLM2_6b ChatGLM3_6b ChatGLM3_6b_32k 
> CodeGeeX2_6b 
> CodeLlama_13b CodeLlama_7b 
> DeciLM-7B 
> InternLM_7b 
> LLama3.1_70b LLama3.1_8b LLama3_70b LLama3_8b 
> Llama2_13b Llama2_70b_chat Llama2_7b Llama2_7b_int4_awq Llama2_7b_int4_gptq Llama2_7b_sql_lora Llama_13b Llama_30b Llama_65b Llama_7b 
> Mistral-7B-v0.1 
> Mixtral-8x7B-V0.1 
> Qwen2_72b Qwen2_7b Qwen_14b_chat Qwen_1_5_14b Qwen_1_5_14b_int8_gptq Qwen_1_5_32b Qwen_1_5_72b Qwen_1_5_7b Qwen_72b_chat Qwen_7b 
> Yi_34b Yi_6b 
> bloom-7b1 
> folcon_40b folcon_7b 
> glm4_9b_chat 
> gpt4all-j gpt_bigcode-santacoder 
> mpt-30b 
> pythia-12b 
> vicuna-13b-v1.5


## vllm目录结构及说明
```

. 📂 vllm
└── 📂 code/
│  └── 📂 opencompass/
│  └── 📂 longbench/
│  └── 📂 src/
│  └── 📂 tools/
│  └── 📂 utils/
│  ├── 📄 bench_test.py
│  ├── 📄 c-eval.py
│  ├── 📄 convert_EAGLE_ckpt_to_vllm_compatible.py
│  ├── 📄 openai_chatcompletion_client.py
│  ├── 📄 openai_completion_client.py
│  ├── 📄 run_benchmark_serving.sh
│  ├── 📄 run_ceval_client.py
│  ├── 📄 run_multimodal.py
│  ├── 📄 run_offline_inference_demo.sh
│  ├── 📄 test_Speculative_Decoding.py
│  ├── 📄 test_prefix_caching.py
└── 📂 dataset/
│  ├── 📄 ShareGPT_V3_unfiltered_cleaned_split.json(需要自己准备，具体下载地址见下文-benchmark_serving)
└── 📂 data/
│  ├── 📄 demo.jpeg
│  ├── 📄 demo.jpg
│  ├── 📄 input_data.txt
└── 📂 models/

# TODO:
code下存放的是测试代码和脚本，dataset是启动openai_api服务端用到的，models下存放的是支持的模型的配置文件，multimodal_test是多模态模型测试。
```



## C-Eval 精度测试
1.  安装依赖（默认已预装）
    ```shell
    pip install lm-eval==0.4.2
    ```

    安装原生的 0.4.2 之后，需要修改 安装lm-eval路径，比如  
    ```/opt/conda/lib/python3.8/site-packages/lm_eval/api/model.py```  **_encode_pair** 方法为下面的： 

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

2.  建立软连接：
    ```shell
    cd /workspace/ModelZoo.LLM.Inference/vllm
    ln -s /pde_ai/datasets/dataset-7/ModelZoo_LLM_data/ceval ceval
    ln -s /pde_ai/datasets/dataset-7/ModelZoo_LLM_data/lm_eval_code/exact_match exact_match
    ```
3. 执行
    ```shell
    # 其中${Model_name}为models中目录名。
    python code/c-eval.py --model ./models/${Model_name}
    ```



## benchmark throughput 性能测试

### 使用方式

```shell
#! 其中${Model_name}为models中目录名。

# 默认跑32条测试数据，默认输入长度1024 默认输出长度 1024
python code/bench_test.py --model ./models/${Model_name}

# 设置跑24条数据测试 ，默认： 输入长度1024 输出长度1024
python code/bench_test.py --model ./models/${Model_name} --num-prompts 24

# 设置跑1024条数据测试，输入长度512，输出长度 128
python code/bench_test.py --model ./models/${Model_name} --num-prompts 1024 --input-len 512 --output-len 128

# 表示进行批次跑，将会一次加载模型跑看护的35个case性能数据
python code/bench_test.py --model ./models/${Model_name} --batched-test

# 如果需要方便的提取TPS，TTFT等数据，将命令修改成如下形式
Log_Name=Anything_u_want && \
CUDA_VISIBLE_DEVICES=${0~7} MX_VLLM_ENABLE_PROFILE=1 python ./code/bench_test.py --model ./models/Qwen2.5_72b_int4_awq --batched-test --num-scheduler-steps 8  \
2>&1 |tee ${Log_Name}.log && awk 'BEGIN {print "case,TPS,TTFT"} /Throughput/ {printf "%s,%f,%f\n", $(NF-14), $(NF-10), $(NF-6)}' ${Log_Name}.log  > TPS_${Log_Name}.csv
```

----  2024.07.31  ----
### 新增环境变量 `MX_VLLM_ENABLE_PROFILE`
*   使能 `MX_VLLM_ENABLE_PROFILE` 环境变量后将会在 `./mx_profile/` 文件夹通过torch_profiler 工具生成csv原始文件（如果跑35个case的话，目前只统计 input_len=256,output_len=128 以及 input_len=1024,output_len=1024 数据）

    > 生成的对应文件夹路径下的csv 可以通过 以下脚本完成 kernel 汇总（注：需要 安装openxl包： `pip install openxl`）  
    > **请在 `vllm` 目录下执行脚本**
    > <!-- TODO: 统计汇总脚本待更新，预计2025.4.30前完成 -->
    > ```shell
    > python ./code/tools/statistics_csv.py ./mx_profiler/
    > ```



## LoRA 特性支持 （Released版本大于等于 2.23）benchmark 

### 离线推理脚本（配置 lora_path 为微调的LoRA模型路径）：

```shell
python code/src/offline_inference_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /pde_ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```

### multi-LoRA 推理脚本：

```shell
python code/src/offline_inference_multi_lora.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /pde_ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```


### LoRA 跑性能数据 --当前性能较差，后续会对其进行优化
<!-- TODO: 随时修改 -->
```shell
python code/bench_test.py --model ./models/Llama2_7b_sql_lora/ --num-prompts 64 --input-len 1024 --output-len 1024
```



## 量化特性支持（Released版本大于等于 2.23）

用法同普通一致，模型路径要为 gptq/awq 模型路径，
```shell
python code/src/offline_inference.py --model /pde_ai/models/llm/quantize_model/llama-2-7b-int4-gptq/
```



## 多模态模型 特性支持（Released版本大于等于 2.25.2）

简易启动命令：
```shell
 python code/run_multimodal.py --model ./models/InternVL-chat-v1.5/
```



## 本地推理demo

1.  脚本为 code/run_offline_inference_demo.sh：
    ```shell
    python src/offline_inference.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf
    ```

2.  运行该脚本：
    ```shell
    bash run_offline_inference_demo.sh
    ```

    > 脚本内需要根据使用模型情况修改模型所在目录，如 --model /pde_ai/models/llm/Llama/Llama-2-7b-hf;  
    > 可根据需要修改code/src/offline_inference.py内的prompts;  
    > 详细参数信息见code/src/offline_inference.py



## benchmark_throughput

1. 脚本为 `./code/src/benchmark_throughput.py`，该脚本可被 `./code/bench_test.py` 调用，也可以自行调用来测试单个case，当然，该脚本提供了 ```--batched-test``` 命令参数，可以用来测试35个case。**需要注意的是，提供 ```--batched-test``` 参数时不可以省略 ```--input-len``` 和 ```--output-len``` 参数**

    ```shell
    python src/benchmark_throughput.py --backend hf --model /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --tokenizer /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --input-len 512 --output-len 128 --hf-max-batch-size 8
    ```



## benchmark_serving

测试 线上服务 的步骤如下

1.  启动在线服务
    执行online benchmark前需要有对应服务启动，简易启动命令: 

    ```shell
    CUDA_VISIBLE_DEVICES=${0~7} vllm serve /pde_ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ -pp 1 -tp 1  --trust-remote-code --dtype bfloat16 --max-model-len 2048 --max-num-batched-tokens 2048 --swap-space 16 --gpu-memory-utilization 0.95 --distributed-executor-backend ray
    ```

    > 1. 请根据需要自行设置 `-pp` 和 `-tp` 参数
    > 2. 必要时请设置 `VLLM_PP_LAYER_PARTITION` 环境变量
    > 3. `--max-num-batched-tokens` 在 GPU 内存足够时，推荐将其设置成与 `--max-model-len` 相同的值，否则测试时的 `--input-len` 为 `min(max_num_batched_tokens, max_model_len)`

3.  接着启动客户端的测试脚本，默认客户端与服务端处于同一网卡的设备中
    1.  新的测试脚本为 `./code/src/benchmark_serving_new.py`
   
        > 1. 新的测试脚本主要改变了一个命令参数以避免歧义
        >    原 `--num-prompts` >> 变为 `--num-samples`
        >    该参数的目的为**设置测试时的样本数量**
        > <br>
        > 2. 如果需要规定测试时的**并发数量**，请设置`--max-concurrency`
        > <br>
        > 3. 需要了解脚本使用详情，请使用 `--help/-h`

        如果需要模拟不同的随机状况下，测试服务器的负载及其他详情，这里推荐一个命令，**其中命令参数值，请根据需要进行更改**

        ```shell
        python code/src/benchmark_serving_new.py --model /pde_ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ --dataset_name random --random_input_len 1024 --random_output_len 1024 --num-samples 500 --trust-remote-code --ignore-eos --max-concurrency 50 --request-rate 0.6
        ```

    2.  旧的测试脚本为 `./code/run_benchmark_serving.sh`，该脚本的内容如下：
    
        ```shell
        python src/benchmark_serving.py --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /pde_ai/models/llm/Llama/Llama-2-7b-hf/ --num-prompts 10
        ```

        > 服务启动后可在脚本内修改参数，详细参数信息见code/src/benchmark_serving.py

        启动简易的测试时，可以执行该脚本：
        ```shell
        bash run_benchmark_serving.sh
        ```



## 启动openai_api服务端 

1.  确保安装了 eval-type-backport 包
    ```shell
    pip install eval-type-backport
    ```

2.  简易启动命令:
    1.  启动在线服务
        ```shell
        vllm serve /pde_ai/models/llm/Llama/Llama-2-7b-hf/ -pp 1 -tp 1  --trust-remote-code --distributed-executor-backend ray --max-model-len 4096 --swap-space 16 --gpu-memory-utilization 0.95
        ```
    
    2.  执行客户端测试，`/workspace/ModelZoo.LLM.Inference/vllm/code`目录下包含completion和chatcompletion两个客户端sample
        ```shell
        python openai_chatcompletion_client.py

        python openai_completion_client.py
        ```

## 测试vllm 0.6.6 APC （Automatic Prefix Caching）

> **参数说明**
> 1. `--model`
>    模型路径

使用pytext 启动 APC 命令
```shell
python code/test_prefix_caching.py --model /pde_ai/models/llm/Llama/Meta-Llama-3-8B-Instruct/
```



## 测试vllm 0.6.6 Chunked Prefill 

* 默认max_num_batched_tokens=512
  
    ```
    # 注意：--model 指的是 模型名称，如果在镜像中运行请确认模型路径是否正确，模型路径配置在：./models/Llama_7b/config.json 中的 model_path 字段。
    python ./code/bench_test.py --model models/Llama_7b  --enable-chunked-prefill
    ```



## Speculative Decoding
> **参数说明**
> 1. --speculative-model
>    speculative 模型的路径，MLP Speculators ，EAGLE based draft models 等。
> 2. --num-speculative-tokens
>    推测令牌 数量，3-10 ，默认值 5
> 3. --ngram-prompt-lk-max
>    lookup  最大数量，默认值4


**注意事项：**
1.  HF 上下载的EAGLE 模型文件不能直接被vllm 使用，可以使用 `convert_EAGLE_ckpt_to_vllm_compatible.py` 转换为vllm 可用的模型文件和配置文件。
    命令如：

    ```shell
    python code/convert_EAGLE_ckpt_to_vllm_compatible.py /pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin /pde_ai/models//llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors
    ```

    > **参数及结果说明：**
    > `/pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin` 为原始模型权重
    > `/pde_ai/models//llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors` 主模型的权重最后一个分片
    > **转换后的文件保存在：** `/pde_ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct_vllm/` 规则  `EAGLE-xxx_vllm`

**示例：**
```shell
python ./code/bench_test.py --model models/LLama3.1_8b  --enable-chunked-prefill --speculative-model /pde_ai/models/llm/Llama/llama3-8b-accelerator  --num-speculative-tokens 4
```

ngram
```shell
python ./code/bench_test.py --model models/LLama3.1_8b  --enable-chunked-prefill --speculative-model "[ngram]" 
```






***

<!-- 自动标题编号 -->
<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2:before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3:before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4:before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5:before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6:before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
</style>