这一页说明如何使用PPL框架推理LLM。

目前PPL已支持的LLM包括Baichuan系列(Baichuan2_7b/Baichuan2_13b)、Chatglm2/3系列(Chatglm2_6b/Chatglm3_6b)、Internlm(Internlm_7b)、Llama系列(Llama_7b/Llama_13b/Llama_30b/Llama_65b/Llama2_13b/Llama2_70b)、Llama3系列(Llama3_8b/Llama3_70b)、Mixtral(Mixtral8x7b)、Qwen系列(Qwen_7b/Qwen_14b/Qwen_72b/Qwen1.5_32b)、Yi系列(Yi1.5_6b-chat)。


## 目录结构及说明

```
├── PPL
|   ├── Code
|   │   ├── src                                                    #    功能依赖文件
|   │   │   ├── baichuan13b
|   │   │   │   ├── ConvertWeightToPMX.py
|   │   │   │   ├── Demo.py
|   │   │   │   └── Export.py
|   │   │   ├── ......
|   │   │   │── qwen
|   │   │   │   ├── ConvertWeightToPMX.py
|   │   │   │   ├── Demo.py
|   │   │   │   └── Export.py
|   │   │   ├── http
|   │   │   │   ├── http_to_grpc.py
|   │   │   │   ├── llm_pb2_grpc.py
|   │   │   │   └── llm_pb2.py
|   │   │   ├── compare.py
|   │   │   └── benchmark.sh
|   │   └── Start.py                                                #   操作命令主入口
|   |
|   ├── Input                                                       #   测试输入数据
|   │   ├── tokens_input_1024
|   │   ├── ...
|   │   └── tokens_input_8
|   |
|   ├── Model                                                       #   模型配置文件及模型生成默认目录
|   │   ├── Baichuan2_13b
|   │   │   ├── Config.json
|   │   │   └── ServiceConfig.json
|   │   ├── ......
|   │   └── Qwen
|   │       ├── Config.json
|   │       └── ServiceConfig.json
|   └── runtest.sh
```

## 环境依赖

    运行PPL相关操作需要完成安装MacaPMX及ppl.llm.serving，Python版本3.8，用到Python标准库os, subprocess, json, sys, tempfile, signal

    此外需要正确配置环境变量（默认在Docker镜像中已设置）：

        export MACA_PATH=your_maca_path
        export PATH=${MACA_PATH}/bin:${PATH}
        export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
        export USE_GEMM_NN=true

## 支持命令（在ModelZoo.LLM.Inference/PPL目录下执行）

    命令结构为:
    ./runtest.sh $YOUR_COMMAND $CONFIG_FILE
                       |            |
                (绝对或相对路径)(绝对或相对路径)

    配置文件已经在代码中给出，每个模型单独对应一个目录，命令相关配置在配置文件内有对应关系。其中，
    $YOUR_COMMAND的可选项包括：convert_to_pmx, split_pmx_model, merge_pmx_model, pmx_model_test, convert_to_onnx, onnx_accuracy_test, onnx_performance_test, start_llm_server, mmlu_accuracy_test; 
    $CONFIG_FILE=Model/${Model_name}/Config.json 或 Model/${Model_name}/ServiceConfig.json, 
    ${Model_name} 可选项为Model目录下所有的文件夹名, 目前已包括：Baichuan2_7b, Baichuan2_13b, Chatglm2_6b, Chatglm3_6b, CodeLlama_7b, Internlm_7b, Llama_7b, Llama_13b, Llama_30b, Llama_65b, Llama2_13b, Llama2_70b, Llama3_8b, Llama3_70b, Mixtral8x7b, Qwen_7b, Qwen_14b, Qwen_72b, Qwen1.5_32b, Yi1.5_6b-chat

## 以Llama3-70B模型为例（在ModelZoo.LLM.Inference/PPL目录下执行）

### 1. 原始模型转为PMX模型
    需要修改 Model/Llama3_70b/Config.json 中 convert_to_pmx 字段下的信息，
    "convert_to_pmx": {
        "origin_model_dir": "/external/models/Llama/Meta-Llama-3-70B/",
        "enable_using_safetensors": true,
        "pmx_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model/"
    }
    origin_model_dir 为原始模型路径；
    enable_using_safetensors 为权重是否是safetensors格式，如果是则设为true，如果不是则设为false，目前仅支持bin和safetensors；
    pmx_model_output_dir 为保存PMX模型路径。

    然后执行：
    ./runtest.sh convert_to_pmx Model/Llama3_70b/Config.json
    将在指定路径生成原始模型的PMX模型。

### 2. PMX模型切分
    注意！14B以下的模型单卡就可以运行，不需要做模型切分和合并，可以略过这一步和第3步。
    需要修改 Model/Llama3_70b/Config.json 中 split_pmx_model 字段下的信息，
    "split_pmx_model": {
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model/",
        "number_of_shards": 4,
        "split_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model-split/"
    }
    pmx_model_dir 为第1步生成的PMX模型的路径；
    number_of_shards 为PMX模型切分数量；
    split_model_output_dir 为保存切分后PMX模型路径。

    然后执行：
    ./runtest.sh split_pmx_model Model/Llama3_70b/Config.json    
    将在指定路径生成4份子模型，以适配多卡并行推理。

### 3. PMX模型合并
    注意！如果没有做第2步模型切分，直接略过这一步！
    与模型切分相反，将多个切分后的PMX子模型合并成一个。大部分情况下不需要做这一步。
    需要修改 Model/Llama3_70b/Config.json 中 merge_pmx_model 字段下的信息，
    "merge_pmx_model": {
        "split_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model-split/",
        "num_of_shards": 4,
        "merged_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model-merged/"
    }
    split_model_dir 为第2步切分后PMX模型路径；
    number_of_shards 为PMX模型切分数量；
    merged_model_output_dir 为保存合并后PMX模型路径。

    然后执行：
    ./runtest.sh merge_pmx_model Model/Llama3_70b/Config.json
    将得到切分后再合并的PMX模型。

### 4. PMX模型测试
    需要修改 Model/Llama3_70b/Config.json 中 pmx_model_test 字段下的信息，
    "pmx_model_test": {
        "num_gpu": 4,
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model-split/",
        "origin_model_tokenizer_path": "/external/models/Llama/Meta-Llama-3-70B/tokenizer.json",
        "seqlen_scale_up": 1,
        "max_gen_len": 256,
        "dump_steps": "0,1,255",
        "dump_tensor_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-dump/",
        "batch_size": 1,
        "cache_layout": 3
    }
    num_gpu 为推理需要的GPU数量，与模型切分数一致；
    pmx_model_dir 为第1步生成的PMX模型的路径；
    origin_model_tokenizer_path 为tokenizer模型目录；
    seqlen_scale_up 为输入字节大小的比例因子；
    max_gen_len 为生成的最大输出长度；
    dump_steps 为保存测试数据的step；
    dump_tensor_path 为保存测试数据的路径；
    batch_size 为batch size大小；
    cache_layout 为cacheAttention中cache存储layout，当前仅支持0和3。

    然后执行：
    ./runtest.sh pmx_model_test Model/Llama3_70b/Config.json
    使用该命令加载PMX模型并执行LLM推理，根据输出结果验证PMX模型转换的正确性。同时，若设置相应dump参数，可以将模型对应step的输入、输出保存下来，作为后续本地部署精度验证的输入及输出参考值。

### 5. 导出为ONNX模型
    在验证PMX模型精度无误后，可执行该命令，将PMX模型导出为ONNX模型！
    需要修改 Model/Llama3_70b/Config.json 中 convert_to_onnx 字段下的信息，
    "convert_to_onnx": {
        "num_gpu": 4,
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-model-split/",
        "origin_model_tokenizer_path": "/external/models/Llama/Meta-Llama-3-70B/tokenizer.json",
        "onnx_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/",
        "cache_layout": 3
    }
    num_gpu 为推理需要的GPU数量，与模型切分数一致；
    pmx_model_dir 为第1步生成的PMX模型的路径；
    origin_model_tokenizer_path 为tokenizer模型目录；
    onnx_model_output_dir 为保存ONNX模型路径；
    cache_layout 为cacheAttention中cache存储layout，当前仅支持0和3。

    然后执行：
    ./runtest.sh convert_to_onnx Model/Llama3_70b/Config.json
    将在指定路径生成ONNX模型。

### 6. ONNX模型精度验证
    需要修改 Model/Llama3_70b/Config.json 中 onnx_accuracy_test 字段下的信息，
    "onnx_accuracy_test": {
        "step": 0,
        "num_gpu": 1,
        "ppl_serving_dir": "/opt/maca-ai/ppl.llm.serving/bin/",
        "test_data_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/pmx-dump/",
        "onnx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/",
        "out_put_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-accuracy-test-result/"
    }
    step 为与PMX模型测试时的step相对应
    num_gpu 为推理需要的GPU数量，与模型切分数一致；
    ppl_serving_dir 为可执行文件pplnn_llm所在目录；
    test_data_dir 为模型输入文件目录，此处使用PMX模型测试时保存的数据；
    onnx_model_dir 为第3步保存ONNX模型路径；
    out_put_dir 为模型输出文件保存目录

    然后执行：
    ./runtest.sh onnx_accuracy_test Model/Llama3_70b/Config.json
    会在终端输出与PMX模型输出的余弦相似度，并以此判定ONNX模型的精度是否正常。

### 7. ONNX模型性能测试
    需要修改 Model/Llama3_70b/Config.json 中 onnx_performance_test 字段下的信息，
    "onnx_performance_test": {
        "model_name": "llama3",
        "ppl_serving_dir": "/opt/maca-ai/ppl.llm.serving/bin/",
        "onnx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/",
        "onnx_model_param_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/params.json",
        "tensor_parallel_size": 4,
        "top_p": 0.0,
        "top_k": 1,
        "temperature": 1.0,
        "warmup_loops": 2,
        "benchmark_loops": 2,
        "input_file_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Input/",
        "input_file_base": "tokens_input",
        "input_token_list": "8",
        "output_token_list": "256",
        "batch_size_list": "1,2,4,8,16,32,64,128,256",
        "do_tracer": false,
        "enable_output_logs": false,
        "log_path": "",
        "output_result_json_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-performance-test-result/result.json"
    }
    model_name 为模型名称；
    ppl_serving_dir 可执行文件benchmark_llama所在目录；
    onnx_model_dir 为第3步保存ONNX模型路径；
    onnx_model_param_path 为ONNX模型的params.json文件路径；
    tensor_parallel_size 与模型切分数量一致；
    top_p、top_k、temperature 为sampling参数；
    warmup_loops 为预热次数；
    benchmark_loops 为性能测试执行次数；
    input_file_dir 为测试输入文件目录；
    input_file_base 为模型输入文件名前缀；
    input_token_list 为模型输入token长度列表；
    output_token_list 为模型生成token长度列表；
    batch_size_list 为性能测试batchsize列表；
    do_tracer 为是否使用mcTrace，通常为False；
    enable_output_logs 为运行日志输出开关；
    log_path 为日志保存路径；
    output_result_json_path 为测试结果保存路径。

    然后执行：
    ./runtest.sh onnx_performance_test Model/Llama3_70b/Config.json

### 8. 服务化部署
    配置文件: Model/Llama3_70b/ServiceConfig.json
    {
        "ppl_serving_dir": "/opt/maca-ai/ppl.llm.serving/bin/",
        "server_config": {
            "model_type": "llama3",
            "model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/",
            "model_param_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama3_70b/onnx-model/params.json",
            "tokenizer_path": "/external/models/Llama/Meta-Llama-3-70B/tokenizer.model",
            "tensor_parallel_size": 4,
            "top_p": 0.0,
            "top_k": 1,
            "quant_method": "none",
            "max_tokens_scale": 0.6,
            "max_tokens_per_request": 4096,
            "max_running_batch": 1024,
            "max_tokens_per_step": 8192,
            "host": "0.0.0.0",
            "port": 23333
        },
        "enable_http_server": true,
        "http_server_config": {
            "host": "0.0.0.0",
            "port": 23334,
            "threads": 50
        }
    }
    ppl_serving_dir 为可执行文件ppl_llm_server所在目录；
    server_config：
        model_dir 为ONNX模型目录；
        model_param_path 为ONNX模型的params.json文件路径；
        tokenizer_path 为tokenizer模型目录，通常在原始模型路径下；
        tensor_parallel_size 与模型切分数量一致；
        top_p， top_k 为sampling需要参数；
        quant_method 为量化方法，默认为none；
        max_tokens_scale 为模型额外占用显存比例，若测试大batchsize，建议将值调高(0.9)；
        max_tokens_per_request 为单次请求的最大token数，建议设置4096；
        max_running_batch 为推理最大batchsize；
        max_tokens_per_step 为单个step处理最大token数；
        host 为grpc服务端host；
        port 为grpc服务端port；
    enable_http_server http服务开关；
    http_server_config：
        host 为http服务端host；
        port 为http服务端port；
        threads 为工作线程数量。

    执行：
    ./runtest.sh start_llm_server Model/Llama3_70b/ServiceConfig.json

    执行该命令将启动大模型服务，默认接收grpc请求，客户端示例代码可在安装ppl.llm.serving后在/opt/maca-ai/ppl.llm.serving/samples/samples/ppl_server_client/目录下找到，包含C++和Python两种版本。python客户端依赖grpc，使用时需安装grpcio、grpcio-tools两个依赖项（默认在Docker镜像中已安装）。
        pip install grpcio
        pip install grpcio-tools
    若选择开启http服务，可接收http和grpc两种请求，http服务需要安装flask、msgpack依赖项（默认在Docker镜像中已安装）。
        pip install flask
        pip install msgpack
    http客户端示例代码可见/opt/maca-ai/ppl.llm.serving/samples/samples/ppl_server_client/python_client/http_client.py，需要安装requests依赖项（默认在Docker镜像中已安装）。
        pip install requests

    若需要输出prefill性能统计数据，设置该环境变量
        export SHOW_PREFILL=true

    当输出下列日志时，表示llama_7b服务启动完成：
        [INFO][2023-12-07 16:05:29.304][llama_worker.cc:962] waiting for request ...

    服务端完成部署，客户端成功发送请求，服务端完成请求的处理并将结果发送至客户端后，服务端会输出当前部署大模型的性能数据。

        [PERF] --- step 1 -------------------------------------------------
        [PERF]  |- memory usage: (63.59 - 16.41) -> 47.18 GiB
        [PERF]  |- kv cache usage: 0.02 %
        [PERF]  |- pending task number: 0
        [PERF]  |- running batch: 1, max running batch: 1
        [PERF]  |- finished query count: 0, QPS: 0.00
        [PERF]  |- gen token count: 1, avg gen len: 0.00, TPS: 8.58
        [PERF]  |- pipeline          | cur: 116.62 ms, | avg: 116.62 ms, | total: 116.62 ms
        [PERF]  |-- batching         | cur: 0.00 ms, | avg: 0.00 ms, | total: 0.00 ms
        [PERF]  |-- copy inputs      | cur: 0.19 ms, | avg: 0.19 ms, | total: 0.19 ms
        [PERF]  |-- model inference  | cur: 116.04 ms, | avg: 116.04 ms, | total: 116.04 ms
        [PERF]  |-- sampling         | cur: 0.37 ms, | avg: 0.37 ms, | total: 0.37 ms
        [PERF]  |-- send response    | cur: 0.01 ms, | avg: 0.01 ms, | total: 0.01 ms
        [PERF]  |-- early finish     | cur: 0.00 ms, | avg: 0.00 ms, | total: 0.00 ms
        [PERF]  |- schedule cost: 0.50 %
        [PERF] --- step 100 -------------------------------------------------
        [PERF]  |- memory usage: (63.59 - 16.41) -> 47.18 GiB
        [PERF]  |- kv cache usage: 0.02 %
        [PERF]  |- pending task number: 0
        [PERF]  |- running batch: 1, max running batch: 1
        [PERF]  |- finished query count: 0, QPS: 0.00
        [PERF]  |- gen token count: 100, avg gen len: 0.00, TPS: 51.21
        [PERF]  |- pipeline          | cur: 16.81 ms, | avg: 19.53 ms, | total: 1952.56 ms
        [PERF]  |-- batching         | cur: 0.00 ms, | avg: 0.00 ms, | total: 0.01 ms
        [PERF]  |-- copy inputs      | cur: 0.09 ms, | avg: 0.07 ms, | total: 7.32 ms
        [PERF]  |-- model inference  | cur: 16.57 ms, | avg: 19.26 ms, | total: 1926.25 ms
        [PERF]  |-- sampling         | cur: 0.14 ms, | avg: 0.17 ms, | total: 16.88 ms
        [PERF]  |-- send response    | cur: 0.01 ms, | avg: 0.02 ms, | total: 1.87 ms
        [PERF]  |-- early finish     | cur: 0.00 ms, | avg: 0.00 ms, | total: 0.00 ms
        [PERF]  |- schedule cost: 1.35 %

    若在启动服务前设置SHOW_PREFILL环境变量，服务端会输出prefill阶段的性能数据，即step1的性能数据；若不设置，则输出100整数倍及最后一个step的性能数据。
    重点性能参数包括：
        memory usage：当前step显存使用情况
        running batch：当前step正在推理的batchsize
        max running batch：从当前step回溯的历史最大推理batchsize
        finished query count：当前step已经结束生成的请求数量
        pipeline：整体流程耗时，包含当前step耗时，平均耗时及总耗时
        model inference：大模型推理的耗时，包含当前step耗时，平均耗时及总耗时

### 9. mmlu 精度测试
    配置文件与上一步的服务化部署相同，具体说明见服务化部署。
    执行：
    ./runtest.sh mmlu_accuracy_test Model/Llama3_70b/ServiceConfig.json

    测试结束后输出如下：    
    Average accuracy 0.912 - world_religions
    Average accuracy 0.501 - math
    Average accuracy 0.643 - health
    Average accuracy 0.728 - physics
    Average accuracy 0.888 - business
    Average accuracy 0.899 - biology
    Average accuracy 0.617 - chemistry
    Average accuracy 0.641 - computer science
    Average accuracy 0.817 - economics
    Average accuracy 0.717 - engineering
    Average accuracy 0.676 - philosophy
    Average accuracy 0.793 - other
    Average accuracy 0.306 - history
    Average accuracy 0.944 - geography
    Average accuracy 0.557 - politics
    Average accuracy 0.887 - psychology
    Average accuracy 0.892 - culture
    Average accuracy 0.111 - law
    Average accuracy 0.650 - STEM
    Average accuracy 0.391 - humanities
    Average accuracy 0.805 - social sciences
    Average accuracy 0.730 - other (business, health, misc.)
    Average accuracy: 0.616
    
    同时会在Model/Llama3_70b/生成results_Llama3_70b文件夹，用以保存各个子项测试结果csv文件。