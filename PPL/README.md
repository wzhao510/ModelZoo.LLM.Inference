## 目录结构及说明

```
.
├── Code
│   ├── src                                                    #    功能依赖文件
│   │   ├── baichuan13b
│   │   │   ├── ConvertWeightToPMX.py
│   │   │   ├── Demo.py
│   │   │   └── Export.py
│   │   ├── ......
│   │   │── qwen
│   │   │   ├── ConvertWeightToPMX.py
│   │   │   ├── Demo.py
│   │   │   └── Export.py
│   │   ├── http
│   │   │   ├── http_to_grpc.py
│   │   │   ├── llm_pb2_grpc.py
│   │   │   └── llm_pb2.py
│   │   ├── compare.py
│   │   └── benchmark.sh
│   └── Start.py                                                #   操作命令主入口
|
├── Input                                                       #   测试输入数据
│   ├── tokens_input_1024
│   ├── ...
│   └── tokens_input_8
|
├── Model                                                       #   模型配置文件及模型生成默认目录
│   ├── Baichuan2_13b
│   │   ├── Config.json
│   │   └── ServiceConfig.json
│   ├── ......
│   └── Qwen
│       ├── Config.json
│       └── ServiceConfig.json
│── README.md
└── runtest.sh
```

## 环境依赖

    运行PPL相关操作需要完成安装MacaPMX及ppl.llm.serving，Python版本3.8，用到Python标准库os, subprocess, json, sys, tempfile, signal

    此外需要正确配置环境变量（默认在Docker镜像中已设置）：

        export MACA_PATH=your_maca_path
        export PATH=${MACA_PATH}/bin:${PATH}
        export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
        export USE_GEMM_NN=true

## 支持命令

    命令结构为 ./runtest.sh YOUR_COMMAND CONFIG_FILE
                    |                        |
                (绝对或相对路径)          (绝对或相对路径)

    配置文件已经在代码中给出，每个模型单独对应一个目录，命令相关配置在配置文件内有对应关系。

## 1、转为PMX模型（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh convert_to_pmx Model/Llama_7b/Config.json

    配置文件说明：
    "convert_to_pmx": {
        "origin_model_dir": "/external/models/Llama_7b",                                                 #   LLM原始权重模型路径
        "enable_using_safetensors": false,                                                          #   原始权重文件格式是否safetensors
        "pmx_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/pmx-model/"  #   输出PMX模型目标路径
    }

    执行该命令，将在指定路径生成原始模型的pmx模型。

## 2、PMX模型切分（以llama_65b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh split_pmx_model Model/Llama_65b/Config.json

    配置文件说明：
    "split_pmx_model": {
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_65b/pmx-model/",                #   待切分PMX模型目录
        "number_of_shards": 4,                                                                               #   PMX模型切分数量
        "split_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_65b/pmx-model-split/"  #   切分后PMX模型保存目录
    }
    
    执行该命令，将在指定路径生成4份子模型，以适配多卡并行推理。

## 3、PMX模型合并（以llama_65b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh merge_pmx_model Model/Llama_65b/Config.json

    配置文件说明：
    "merge_pmx_model": {
        "split_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_65b/pmx-model-split/",            #   待合并PMX模型目录
        "num_of_shards": 4,                                                                                      #   PMX子模型数量
        "merged_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_65b/pmx-model-merged/"    #   合并后PMX模型保存目录
    }

    该命令与模型切分相反，将多个切分后的PMX子模型合并成一个。

## 4、PMX模型测试（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh pmx_model_test Model/Llama_7b/Config.json

    配置文件说明：
    "pmx_model_test": {
        "num_gpu": 1,                                                                               #   模型推理需要的GPU数量，与模型份数相关
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/pmx-model/",        #   PMX模型目录
        "origin_model_tokenizer_path": "/external/models/Llama_7b",                                       #   tokenizer模型目录
        "seqlen_scale_up": 1,                                                                       #   输入字节大小的比例因子
        "max_gen_len": 256,                                                                         #   生成的最大输出长度
        "dump_steps": "0,1,255",                                                                    #   保存测试数据的step
        "dump_tensor_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/pmx-dump/",      #   保存测试数据的路径
        "batch_size": 1,                                                                            #   批处理的数据大小
        "cache_layout": 3                                                                           #   cacheAttention中cache存储layout，当前仅支持0和3。
    }

    当生成PMX模型后，使用该命令加载PMX模型并执行LLM推理，根据输出结果验证PMX模型转换的正确性。同时，若设置相应dump参数，可以将模型对应step的输入、输出保存下来，作为后续本地部署精度验证的输入及输出参考值。

## 5、导出为ONNX模型（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh convert_to_onnx Model/Llama_7b/Config.json

    配置文件说明：
    "convert_to_onnx": {
        "num_gpu": 1,                                                                                   #   模型推理需要的GPU数量
        "pmx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/pmx-model/",            #   PMX模型目录
        "origin_model_tokenizer_path": "/external/models/Llama_7b",                                           #   tokenizer模型目录
        "onnx_model_output_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/",   #   ONNX模型导出目录
        "cache_layout": 3                                                                               #   cacheAttention中cache存储layout，当前仅支持0和3。
    }

    在验证PMX模型精度无误后，可执行该命令，将PMX模型导出为ONNX模型。

## 6、ONNX模型精度验证（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh onnx_accuracy_test Model/Llama_7b/Config.json

    配置文件说明：
    "onnx_accuracy_test": {
        "step": 0,                                                                                          #   与PMX模型测试时的step相对应
        "num_gpu": 1,                                                                                       #   模型推理需要的GPU数量
        "pplnn_llm_dir": "/opt/maca-ai/ppl.llm.serving/bin/",                                               #   可执行文件pplnn_llm所在目录
        "test_data_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/pmx-dump/",                 #   模型输入文件目录，此处使用PMX模型测试时保存的数据
        "onnx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/",              #   ONNX模型目录
        "out_put_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-accuracy-test-result/"   #   模型输出文件保存目录
    }

    大多数情况下，大模型会依托服务端部署提供服务端接口供客户端调用，但在服务化部署前，需要依托本地模型部署进行推理验证，以确认模型精度是否符合预期。该命令将执行ONNX模型精度验证操作，输出数据将与PMX模型测试输出数据进行对比。

## 7、ONNX模型性能测试（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh onnx_performance_test Model/Llama_7b/Config.json

    配置文件说明：
    "onnx_performance_test": {
        "model_name": "llama_7b",                                                                                                   #   模型名称
        "ppl_serving_dir": "/opt/maca-ai/ppl.llm.serving/bin/",                                                                     #   可执行文件benchmark_llama所在目录
        "onnx_model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/",                                      #   ONNX模型目录
        "onnx_model_param_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/params.json",                    #   ONNX模型的params.json文件路径
        "tensor_parallel_size": 1,                                                                                                  #   与模型切分数量一致
        "top_p": 0.0,                                                                                                               #   ```
        "top_k": 1,                                                                                                                 #       推理参数
        "temperature": 1.0,                                                                                                         #   ```
        "warmup_loops": 2,                                                                                                          #   warmup执行次数
        "benchmark_loops": 2,                                                                                                       #   性能测试执行次数
        "input_file_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Input/",                                                          #   模型输入文件目录
        "input_file_base": "tokens_input",                                                                                          #   模型输入文件基础文件名
        "input_token_list": "8,256",                                                                                                #   模型输入token长度列表
        "output_token_list": "256,512",                                                                                             #   模型生成token长度列表
        "batch_size_list": "1,2,4,8,16,32,64,128,256",                                                                              #   性能测试batchsize列表
        "do_tracer": false,                                                                                                         #   使用mcTracer
        "enable_output_logs": false,                                                                                                #   运行日志输出开关
        "log_path": "",                                                                                                             #   日志保存路径
        "output_result_json_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-performance-test-result/result.json" #   测试结果保存路径
    }

    执行该命令，可在本地进行性能测试，为后续服务端部署做性能指标参照。性能测试过程中将遍历batch_size、input_token和output_token进行组合，对每种组合分别做测试，测试结束后会输出对应的性能测试数据，如下所示：
        CSV format header:prefill(ms),decode(ms),avg(ms),tps(ms),mem(gib)
        CSV format output:40.16,14.2576,14.4599,69.1567,14.3461

## 8、服务化部署（以llama_7b为例，在ModelZoo.LLM.Inference/PPL目录下执行）

    ./runtest.sh start_llm_server Model/Llama_7b/ServiceConfig.json

    配置文件说明：
    "ppl_serving_dir": "/opt/maca-ai/ppl.llm.serving/bin/",                                                     #   可执行文件ppl_llm_server所在目录
    "server_config": {
        "model_type": "llama",                                                                                  #   ppl_llm_server支持的模型框架类型，当前支持的模型已经配置完成，不需要做修改
        "model_dir": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/",                       #   ONNX模型目录
        "model_param_path": "/workspace/ModelZoo.LLM.Inference/PPL/Model/Llama_7b/onnx-model/params.json",     #   ONNX模型的params.json文件路径
        "tokenizer_path": "/external/models/Llama_7b/tokenizer.model",                                               #   tokenizer模型目录
        "tensor_parallel_size": 1,                                                                              #   与模型切分数量一致
        "top_p": 0.0,                                                                                           #   推理参数
        "top_k": 1,                                                                                             #   推理参数
        "quant_method": "none",
        "max_tokens_scale": 0.6,                                                                                #   模型额外占用显存比例，若测试大batchsize，建议将值调高(0.9)
        "max_tokens_per_request": 4096,                                                                         #   单次请求的最大token数，建议设置4096
        "max_running_batch": 1024,                                                                              #   执行推理最大batchsize
        "max_tokens_per_step": 8192,                                                                            #   单个step处理最大token数
        "host": "0.0.0.0",                                                                                      #   grpc服务端host
        "port": 23333                                                                                           #   grpc服务端port
    },
    "enable_http_server": true,                                                                                 #   http服务开关
    "http_server_config": {                                                                                     #   http服务端host、port、工作线程数量
        "host": "0.0.0.0",
        "port": 23334,
        "threads": 50
    }

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

## 9、mmlu 精度测试（以chatglm3_6b为例，在ModelZoo.LLM.Inference/PPL目录下执行）
本测试需要使用mmlu数据集，请在测试前从https://huggingface.co/datasets/lighteval/mmlu/blob/main/data.tar下载，解压后请将val和test文件夹拷贝至./Input/mmlu路径。

    ./runtest.sh mmlu_accuracy_test Model/Chatglm3_6b/ServiceConfig.json

    配置文件与服务化部署相同，具体说明见服务化部署，测试结束后输出如下：
        Average accuracy 0.696 - world_religions
        Average accuracy 0.378 - math
        Average accuracy 0.580 - health
        Average accuracy 0.466 - physics
        Average accuracy 0.776 - business
        Average accuracy 0.681 - biology
        Average accuracy 0.488 - chemistry
        Average accuracy 0.524 - computer science
        Average accuracy 0.551 - economics
        Average accuracy 0.503 - engineering
        Average accuracy 0.480 - philosophy
        Average accuracy 0.645 - other
        Average accuracy 0.710 - history
        Average accuracy 0.788 - geography
        Average accuracy 0.747 - politics
        Average accuracy 0.668 - psychology
        Average accuracy 0.732 - culture
        Average accuracy 0.493 - law
        Average accuracy 0.479 - STEM
        Average accuracy 0.530 - humanities
        Average accuracy 0.671 - social sciences
        Average accuracy 0.630 - other (business, health, misc.)
        Average accuracy: 0.573
    同时会在Model/Chatglm3_6b/生成results_Chatglm3_6b文件夹，用以保存各个子项测试结果csv文件。
