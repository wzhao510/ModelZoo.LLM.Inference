# VLLM 项目教程及说明

***
> 这一页说明如何使用vLLM框架推理LLM。建议使用vllm Docker环境，创建容器时添加模型目录映射：-v /pde_ai:/pde_ai，测试工程需放在/workspace下。
> 
> 目前支持模型范围与社区版本同步，相同release版本社区版本支持的模型，我们全部支持
>


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

code下存放的是测试代码和脚本，dataset是启动openai_api服务端用到的，models下存放的是支持的模型的配置文件。
```


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



## benchmark_throughput 本地离线性能测试

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



## benchmark_serving 在线性能测试

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
    #大语言模型使用以下命令进行性能测试
    vllm bench serve --model /external/ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ --dataset_name random --random_input_len 1024 --random_output_len 1024 --num-prompts 320 --trust-remote-code --ignore-eos --max-concurrency 32

    #多模态模型使用以下命令进行性能测试
    vllm bench serve --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-VL-8B-Instruct/ --dataset-name random-mm --backend openai-chat --endpoint /v1/chat/completions --trust-remote-code --ignore-eos  --max-concurrency 32 --num-prompts 128 --port ${port} --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0 --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' --random-mm-bucket-config '{(1080, 1920, 1): 1.0}' --random-input-len 512 --random-output-len 256
    ```
    
## 精度评测

精度评测，我们建议使用opencompass/lm_eval/evalscope 等主流大模型精度评测工具的server api 模式，具体方法参考对应工具的官方文档即可。


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
