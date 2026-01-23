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

code下存放的是测试代码和脚本，dataset是启动openai_api服务端用到的，models下存放的是支持的模型的配置文件，multimodal_test是多模态模型测试。
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
CUDA_VISIBLE_DEVICES=${0~7} python ./code/bench_test.py --model ./models/Qwen2.5_72b_int4_awq --batched-test --num-scheduler-steps 8  \
2>&1 |tee ${Log_Name}.log && awk 'BEGIN {print "case,TPS,TTFT"} /Throughput/ {printf "%s,%f,%f\n", $(NF-14), $(NF-10), $(NF-6)}' ${Log_Name}.log  > TPS_${Log_Name}.csv
```


## LoRA 特性支持 （Released版本大于等于 2.23）benchmark 

### 离线推理脚本（配置 lora_path 为微调的LoRA模型路径）：

```shell
python code/src/offline_inference_lora.py --model /external/ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /external/ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```

### multi-LoRA 推理脚本：

```shell
python code/src/offline_inference_multi_lora.py --model /external/ai/models/llm/Llama/Llama-2-7b-hf/ --lora_path /external/ai/models/models-7/LoRA/lora_test/lora_llama-2-7b/llama-2-7b-sql-lora-test/
```


### LoRA 跑性能数据 --当前性能较差，后续会对其进行优化
```shell
python code/bench_test.py --model ./models/Llama2_7b_sql_lora/ --num-prompts 64 --input-len 1024 --output-len 1024
```



## 量化特性支持（Released版本大于等于 2.23）

用法同普通一致，模型路径要为 gptq/awq 模型路径，
```shell
python code/src/offline_inference.py --model /external/ai/models/llm/quantize_model/llama-2-7b-int4-gptq/
```



## 多模态模型 特性支持（Released版本大于等于 2.25.2）

简易启动命令：
```shell
 python code/run_multimodal.py --model ./models/InternVL-chat-v1.5/
```

多模态模型使用benchmark_serving进行在线性能测试：

    起模型服务 vllm serve ...(具体参考benchmark_serving章节)

    cd ModelZoo.LLM.Inference/vllm/code/src目录下执行以下命令

    ```shell
    python benchmark_serving.py --model {model_name} --dataset-name custom_multiModal --dataset-path {picture_dir}  --trust-remote-code --ignore-eos --backend openai-chat --endpoint /v1/chat/completions  --max-concurrency 32 --num-prompts 128 --custom-input-len 512 --custom-output-len 256 --resize {x,y} 

    注意： 此命令从下个版本开始不再支持，后续请使用下面命令测试

    vllm bench serve --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-VL-8B-Instruct/ --dataset-name random-mm --backend openai-chat --endpoint /v1/chat/completions --trust-remote-code --ignore-eos  --max-concurrency 32 --num-prompts 128 --port ${port} --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0 --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' --random-mm-bucket-config '{(1080, 1920, 1): 1.0}' --random-input-len 512 --random-output-len 256

    或者

    python benchmark_serving.py --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-VL-8B-Instruct/ --dataset-name random-mm --backend openai-chat --endpoint /v1/chat/completions --trust-remote-code --ignore-eos  --max-concurrency 32 --num-prompts 128 --port ${port} --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0 --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' --random-mm-bucket-config '{(1080, 1920, 1): 1.0}' --random-input-len 512 --random-output-len 256

    ```

    其中picture_dir为存放图片的目录，resize为设置图片尺寸若不配置默认为1920,1080


## 本地推理demo

1.  脚本为 code/run_offline_inference_demo.sh：
    ```shell
    python src/offline_inference.py --model /external/ai/models/llm/Llama/Llama-2-7b-hf
    ```

2.  运行该脚本：
    ```shell
    bash run_offline_inference_demo.sh
    ```

    > 脚本内需要根据使用模型情况修改模型所在目录，如 --model /external/ai/models/llm/Llama/Llama-2-7b-hf;  
    > 可根据需要修改code/src/offline_inference.py内的prompts;  
    > 详细参数信息见code/src/offline_inference.py



## benchmark_throughput

1. 脚本为 `./code/src/benchmark_throughput.py`，该脚本可被 `./code/bench_test.py` 调用，也可以自行调用来测试单个case，当然，该脚本提供了 ```--batched-test``` 命令参数，可以用来测试35个case。**需要注意的是，提供 ```--batched-test``` 参数时不可以省略 ```--input-len``` 和 ```--output-len``` 参数**

    * 如若执行的是普通推理模型，可以使用如下参数设置

        ```shell
        python src/benchmark_throughput.py --backend hf --model /external/ai/models/llm/Llama/Llama-2-7b-hf/ --tokenizer /external/ai/models/llm/Llama/Llama-2-7b-hf/ --input-len 512 --output-len 128 --num-prompts 8
        ```
    * 如若执行的是embed模型，可以使用如下设置
        > **注意**，执行该类模型时，num-scheduler-steps必须设置为1，否则会导致报错。
        > 同时为该类模型设置的output-len需要满足模型的规定，如果提醒dimension有问题，请按照log要求进行修改

        ```shell
        python ./code/src/benchmark_throughput.py --model /external/ai/models/llm/jinaai/jina-embeddings-v3/ --batched-test --backend=vllm --max-model-len 2048 --num-prompts 8 --trust-remote-code --dtype float16 --input-len 1024 --output-len 1024 --tensor-parallel-size 1 --num-scheduler-steps 1 --task embed
        ```

    > 这里仅说明常用的命令参数，具体的参数意义或者有哪些命令参数可以使用，请执行 `--help/-h` 查看

    > `--input-len` 输入参数的长度

    > `--output-len` 输出参数的长度

    > `--num-prompts` 即batch-size (由于进行的测试只有一个batch，在该脚本中可以简单理解成样本量)

    > `--max-model-len` 允许的 (input + output) 的最大值

    > `--batched-test` 可以用来测试35个case，**目前还需要提供--input-len和--output-len *未来会进行改进***

    > `--num-scheduler-steps` 通常设置为8，默认值为1，可以提升模型推理的性能

    > `--enable-profile` 开启torch profile，抓取kernel信息

    > `--task` 选择任务类型，目前支持`generate`, `auto`, `embed`和`embedding`，默认值为`auto`



## benchmark_serving

测试 线上服务 的步骤如下

1.  启动在线服务
    执行online benchmark前需要有对应服务启动，简易启动命令: 

    ```shell
    CUDA_VISIBLE_DEVICES=${0~7} vllm serve /external/ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ -pp 1 -tp 1  --trust-remote-code --dtype bfloat16 --max-model-len 2048 --max-num-batched-tokens 2048 --swap-space 16 --gpu-memory-utilization 0.95 --distributed-executor-backend ray -O {"full_cuda_graph": true}
    ```

    > 1. 请根据需要自行设置 `-pp`，`-tp`，`-dp` 参数
    > 3. `--max-num-batched-tokens` 在 GPU 内存足够时，推荐将其设置成与 `--max-model-len` 相同的值，否则测试时的 `--input-len` 为 `min(max_num_batched_tokens, max_model_len)`

3.  接着启动客户端的测试脚本，默认客户端与服务端处于同一网卡的设备中
    测试脚本为 `./code/src/benchmark_serving.py`
   
        > 1. 如果需要规定测试时的**并发数量**，请设置`--max-concurrency`
        
        > 2. 需要了解脚本使用详情，请使用 `--help/-h`

        如果需要模拟不同的随机状况下，测试服务器的负载及其他详情，这里推荐一个命令，**其中命令参数值，请根据需要进行更改**, 这里建议num_prompts 配置为 max_concurrency的10倍。

    ```shell
    python code/src/benchmark_serving.py --model /external/ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ --dataset_name random --random_input_len 1024 --random_output_len 1024 --num-prompts 320 --trust-remote-code --ignore-eos --max-concurrency 32

    或者使用下面命令

    vllm bench serve --model /external/ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ --dataset_name random --random_input_len 1024 --random_output_len 1024 --num-prompts 320 --trust-remote-code --ignore-eos --max-concurrency 32
    ```
    

## 启动openai_api服务端 
1.  确保安装了 eval-type-backport 包
    ```shell
    pip install eval-type-backport
    ```

2.  简易启动命令:
    1.  启动在线服务
        ```shell
        vllm serve /external/ai/models/llm/Llama/Llama-2-7b-hf/ -pp 1 -tp 1  --trust-remote-code --distributed-executor-backend ray --max-model-len 4096 --swap-space 16 --gpu-memory-utilization 0.95 -O {"full_cuda_graph": true}
        ```
    
    2.  执行客户端测试，`/workspace/ModelZoo.LLM.Inference/vllm/code`目录下包含completion和chatcompletion两个客户端sample
        ```shell
        python openai_chatcompletion_client.py

        python openai_completion_client.py
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
    python code/convert_EAGLE_ckpt_to_vllm_compatible.py /external/ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin /external/ai/models//llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors
    ```

    > **参数及结果说明：**
    > `/external/ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct/pytorch_model.bin` 为原始模型权重
    > `/external/ai/models/llm/Qwen/Qwen2-7B-Instruct/model-00004-of-00004.safetensors` 主模型的权重最后一个分片
    > **转换后的文件保存在：** `/external/ai/models/llm/Qwen/EAGLE-Qwen2-7B-Instruct_vllm/` 规则  `EAGLE-xxx_vllm`

**示例：**

```shell
python ./code/bench_test.py --model models/LLama3.1_8b  --enable-chunked-prefill --speculative-model /external/ai/models/llm/Llama/llama3-8b-accelerator  --num-speculative-tokens 4
```

ngram
```shell
python ./code/bench_test.py --model models/LLama3.1_8b  --enable-chunked-prefill --speculative-model "[ngram]" 
```



## opencompass精度评测
测试相关代码和数据集路径：/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/opencompass/

1. opencompass环境准备
pip install opencompass==0.4.2
pip install math_verify latex2sympy2_extended

2. serve启动(以Deepseek R1为例，推荐max-model-len为20K)
vllm serve /mnt/deepseek/DeepSeek-R1-BF16/ -tp 8  -pp 4 --trust-remote-code --distributed-executor-backend ray --dtype bfloat16 --max-model-len 20480 --swap-space 16 --gpu-memory-utilization 0.95 --max-num-batched-tokens 20480

3. ceval精度测试(只能离线使用)
    a. 替换./ceval_gen_5f30c7.py至/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/ceval/ceval_gen_5f30c7.py, 注意95行需要{opencompass完整路径}
    b. 修改/opt/conda/lib/python3.10/site-packages/opencompass/datasets/ceval.py        #line 25
        for split in ['val']:
    c. 修改./eval_ceval.py中模型路径和serve端口号
    d. run
        opencompass ./eval_ceval.py

4. mmlu精度测试
    a. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 2
        from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
    b. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 104
        retriever=dict(type=ZeroRetriever),
    c. 修改./eval_ceval.py中模型路径和serve端口号
    d1. 在线run
        opencompass ./eval_mmlu.py
 
    d2. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 116
        path='{opencompass完整路径}/data/mmlu_csv/'
    e. 替换./mmlu.py至/opt/conda/lib/python3.10/site-packages/opencompass/datasets/mmlu.py
    f. 离线run
        opencompass ./eval_mmlu.py

    ps: qwen3系列mmlu精度测试需要额外修改prompt提示词

        修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 78

            _hint = f'There is a single choice question about {_name.replace("_", " ")}. Answer the question by replying Answer: A, Answer: B, Answer: C or Answer: D.'

        修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 86

            f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n'

        修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py        #line 98

            prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n'



5. gsm8k精度测试
    a. 修改./eval_gsm8k.py中模型路径和serve端口号
    b1. 在线run
        opencompass ./eval_gsm8k.py

    b2. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/gsm8k/gsm8k_gen_1dce88.py        #line 81
        path='{opencompass完整路径}/data/gsm8k_jsonl/main/',
    c. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/gsm8k/gsm8k_gen_1dce88.py        #line 70
        inferencer=dict(type=GenInferencer))
    d. 修改/opt/conda/lib/python3.10/site-packages/opencompass/datasets/gsm8k.py            #line 27
        split_path = os.path.join(path, split + '-00000-of-00001.jsonl')
    e. 离线run
        opencompass ./eval_gsm8k.py
 
6. math500精度测试
    a. 修改./eval_math.py中模型路径和serve端口号
    b1. 在线run
        opencompass ./eval_math.py
 
    b2. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/math/math_500_gen.py        #line 34

          path='{opencompass完整路径}/data/',

    c. 离线run

        opencompass ./eval_math.py

7. aime2024精度测试
    a. 修改./eval_aime2024.py中模型路径和serve端口号
    b1. 在线run
        opencompass ./eval_aime2024.py
 
    b2. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/aime2024/aime2024_gen_6e39a4.py         #line 23

          inferencer=dict(type=GenInferencer)

    c.  修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/aime2024/aime2024_gen_6e39a4.py         #line 34

        path='/mnt/dataset/share/dbshi/opencompass/data/aime.jsonl',

    d. 修改/opt/conda/lib/python3.10/site-packages/opencompass/datasets/aime2024.py        #line 21

        origin_prompt = line['Problem']

    e. 修改/opt/conda/lib/python3.10/site-packages/opencompass/datasets/aime2024.py        #line 23

        line['answer'] = line['Answer']

    f. 离线run

       opencompass ./eval_aime2024.py



8. gpqa精度测试

    a. 修改./eval_gpqa.py中模型路径和serve端口号

    b1. 在线run
        opencompass ./eval_gpqa.py



    b2. 修改/opt/conda/lib/python3.10/site-packages/opencompass/configs/datasets/gpqa/gpqa_openai_simple_evals_gen_5aeece.py        #line 47

        path='{opencompass完整路径}/data/gpqa/',

    c. 离线run

        opencompass ./eval_gpqa.py

## lm_eval 精度评测

1. lm_eval 环境准备

- 安装lm_eval
```shell
从 https://github.com/EleutherAI/lm-evaluation-harness 获取源码

cd lm-evaluation-harness/
pip install -e .
pip install tenacity #mmmu 最小依赖项
```
2. MMMU 多模态数据集精度评测
- 使用离线数据集，配置数据集路径
```shell
修改安装路径 lm_eval\tasks\mmmu\_template_yaml 中 dataset_path为本地路径
```
- 使用在线数据集，无需上述操作，保持默认配置，会自动从互联网获取数据集

3. 精度评测步骤（以Server-Client模式为例）
- 启动vllm server
``` shell
vllm serve $MODEL-PATH   --trust-remote-code   --max-model-len 20480 --max_num_batched_tokens 8192 --gpu-memory-utilization 0.95   -tp $tp_size --served-model-name $your_served_name --port $port
```
- 另开一个终端，执行精度测试客户端命令（以MMMU为例）
``` shell
lm_eval --model local-chat-completions --tasks mmmu_val --model_args model=$your_served_name,base_url= http://localhost:$port/v1/completions,num_concurrent=32,max_retries=2,tokenized_requests=False,tokenizer_backend=None,max_length=16384,max_gen_toks=4096  --apply_chat_template --batch_size 1 --log_samples --output_path $your_output_log_path 
```
## 测试3种mtp方式下模型的功能

```
apt install curl
```
```
export MTP_LOG_PATH=logs/dailytest
bash test_mtp.sh /mxstorage/pde_ai/models/llm/Qwen/Qwen3-14B/ 2 /pde_ai/models/llm/Qwen/Qwen3-14B_eagle3/ 0.85
```
```
#参数说明
# gpu利用率 默认0.85,可省略
bash test_mtp.sh ${model_path} ${gpus_required} ${draft_model_path} ${gpu-memory-utilization} 
```
```
#输出示例
log路径: logs/dailytest
===================================================
model: Qwen3-14B, tp: 2, draft: Qwen3-14B_eagle3  ---> Qwen3-14B.log
开始测试
程序正常测试完成
======== ngrams 测试结果 ========
Prompt: '144的平方根是多少?请直接给出结果不用解释。', Generated text: ' 12 或 -12\n\n嗯，好的，我现在要解决的问题'
测试成功，精度正常

=================================
======== eagle3 测试结果 ========
Prompt: '144的平方根是多少?请直接给出结果不用解释。', Generated text: ' 12或-12\n\n好的，我现在要解决的问题是求'
测试成功，精度正常

=================================
======== draft 测试结果 ========
Prompt: '144的平方根是多少?请直接给出结果不用解释。', Generated text: ' 12或-12\n\n好的，我现在要解决的问题是求'
测试成功，精度正常

=================================
==========================================================
 测试全部完成！
Qwen3-14B正确率：100.00%
==========================================================
```

## 测试tp+dp 不同组合下模型的功能

```
apt install jq
apt install curl
```
```
bash test_tp+dp.sh /mxstorage/pde_ai/models/llm/Qwen/Qwen3-30B-A3B.w8a8/ 2
```
```
#参数
# gpu利用率 默认0.85,可省略
bash test_tp+dp.sh ${model_path} ${least_gpus_required} ${gpu-memory-utilization} 
```
```
#输出示例
本轮测试完成 (5/5)

==========================================================
 最终测试汇总
==========================================================
配置描述              | 正确数  | 总题数  | 正确率
----------------------------------------------------------
TP=2, DP=2 (4 GPU)        | 5          | 5          | 100      %
==========================================================
 测试全部完成！
Qwen3-30B-A3B.w8a8正确率：100%
==========================================================
```


## 测试tp+dcp模型功能
```
apt install jq
apt install curl
```
```
bash test_dcp.sh /mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ 4
```
```
# 参数说明
# gpu利用率 默认0.85,可省略
bash test_dcp.sh ${model_path} ${gpus_required} ${gpu-memory-utilization} 
```
```
# 输出示例
本轮测试完成 (2/5)

==========================================================
 最终测试汇总
==========================================================
配置描述              | 正确数  | 总题数  | 正确率
----------------------------------------------------------
TP=4, DCP=2 (4 GPU)       | 2          | 5          | 40       %
TP=4, DCP=4 (4 GPU)       | 2          | 5          | 40       %
==========================================================
 测试全部完成！
DeepSeek-V2-Lite正确率：40%
==========================================================
```
## lmcache
```
python offline_lmcache_offload.py \
 --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-14B/ \ 
 --port 9001 --test-cpu  --test-disk
```
```
# 参数说明
--model MODEL                                       vLLM 模型路径
--host HOST                                         KVCache 测试服务的主机
--port PORT                                         KVCache 测试服务的端口
--max-model-len MAX_MODEL_LEN                       vLLM 实例的最大模型长度
--test-prompt-tokens TEST_PROMPT_TOKENS             长上下文的 token 数量
--tensor_parallel_size /-tp TENSOR_PARALLEL_SIZE    tp
--test-cpu                                          测试CPU offload
--test-disk                                         测试Disk offload
--log-dir LOG_DIR                                   Server日志文件保存目录
--gpu-memory-utilization                            gpu利用率 默认0.85
```
```
# 输出示例
================================================================================
测试结果汇总
================================================================================
CPU KVCache offload: ✅ 通过
Disk KVCache offload: ✅ 通过

测试完成：通过率：100.0% (2/2)
```
## toolcall
```
python test_toolcall.py --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-14B/ --port 9002 -tp 1
```
```
# 参数说明
--model MODEL                           模型路径
--port PORT                             端口号
--tensor_parallel_size, -tp             tp
--log LOG                               server log 路径
--tool-call-parser TOOL_CALL_PARSER     会根据模型类型去自动选择，也可以强制指定
--disable-auto-tool-choice              关闭tool call
--gpu-memory-utilization                gpu利用率 默认0.85
```
```
# 输出示例
==================================================
Test Report
==================================================

Detailed results:
T01. Math calculation: PASS
T02. Complex calculation: PASS
T03. Weather query: PASS
T04. English weather: PASS
T05. Weather with unit: PASS
T06. Force tool call: PASS
T07. No tool needed: PASS

Conclusion: PASS - Tool call functionality is working
Total tests: 7
Successful: 7
正确率: 100.0%
```

## BAAI rerank 模型精度测试
```
apt install jq
apt install curl
```
```
bash test_rerank.sh /mxstorage/pde_ai/models/llm/BAAI/bge-reranker-v2-m3/
bash test_rerank.sh /mxstorage/pde_ai/models/llm/BAAI/bge-reranker-large/
```
```
# 参数说明
--model MODEL                           模型路径
# 输出示例
==========================================================
bge-reranker-large 精度异常
bge-reranker-large 正确率：0%
==========================================================
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
