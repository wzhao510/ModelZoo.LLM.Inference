# ModelZoo SGLang ReadMe

# 1 目录结构

```markdown
sgl
├──code/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── master.py
│   │   ├── output.py
│   │   ├── slave.py
│   │   ├── task.py
│   ├──utils/
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── bench_sglang.py
│   ├── run_ceval_client.py
├── dataset/
│   └── ceval_val_cmcc.jsonl
├── models/
│   ├── 模型名称
│   │   ├── benchmark.json
│   │   ├── acc.json
│   ├── mechines.json
└── README.md
```

code下存放的是测试代码和脚本, models目录下存放的是支持的模型的配置文件

# 2 测试类型

| 测试类型     | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| benchmark    | 可以组合各种参数的benchmark测试                              |
| acc          | 精度测试，目前支持 mmlu和ceval                               |

## 2.1 服务器信息配置 (mechines.json)

包含节点信息和通用环境变量配置

- **machine_info：节点配置**

| 参数    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| ip      | IP地址                                                       |
| ifname  | 网络接口名，填写与ip匹配的网卡的名。可以在host机器上使用命令ifconfig，然后查看ip对应的网卡名称|
| ib_hcas | 主机通道适配器<br>在host机器上使用ibstat命令来获取，ibstat 输出类似如下结果<pre style="font-size: 12px; line-height: 1.2;">root@****:~# ibstat<br>CA 'mlx5_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                Base lid: 0<br>                ...<br>CA 'mlx5_1'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                ...<br>CA 'mlx5_bond_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 100<br>                ...<br>...</pre>请收集所有名称符合 mlx5_\[数字\]，并且State为Active的结果。<br>如上面，则应设置为 "ib_hcas":"mlx5_0,mlx5_1"， bond_0不需要带上。 |

- **common_envs：通用环境变量配置**

| 参数                                              | 说明                                      |
| ------------------------------------------------- | ----------------------------------------- |
| MACA_SMALL_PAGESIZE_ENABLE                        | 页面大小优化                              |
| TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP | Triton 编译器优化                         |
| TRITON_ENABLE_MACA_CHAIN_DOT_OPT                  | Triton 编译器的链式 Dot 操作优化          |
| MACA_DIRECT_DISPATCH                              | 开启 direct dispatch 功能                     |
| MCDBG_GRAPH_LAUNCH_QUEUE_POLICY                   | 设置 graph 内部创建的 stream/queue 的优先级为high                     |
| MACA_GRAPH_LAUNCH_QUEUE_POLICY                    | 设置 graph 内部创建的 stream/queue 的优先级为high                     |

已有配置默认路径在sgl/models/mechines.json：

```json
{
  "machine_info": [
    {
        "ip": "192.168.3.54",
        "ifname":"bond0",
        "ib_hcas": "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
    },
    {
        "ip": "192.168.3.77",
        "ifname":"bond0",
        "ib_hcas": "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
    }
  ],
  "common_envs": {
    "MACA_SMALL_PAGESIZE_ENABLE": "1",
    "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
    "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
    "MACA_DIRECT_DISPATCH": "1",
    "MCDBG_GRAPH_LAUNCH_QUEUE_POLICY":"3",
    "MACA_GRAPH_LAUNCH_QUEUE_POLICY":"3"
  }
}
```

## 2.2 测试通用配置说明

- **tasks: 测试任务信息**

| 参数        | 说明                         |
| ----------- | ---------------------------- |
| task_name   | 任务名称                     |
| launch_mode | 模式：online、offline        |
| world_size  | 跑本任务测试需要用到的卡数量 |
| server_port | 端口号                       |

- **environment：任务独有环境变量配置**

| 参数                                              | 说明                                      |
| ------------------------------------------------- | ----------------------------------------- |

默认为空，可添加此任务的独有环境变量

- **launch_server：服务配置**

| 参数                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| command_base         | 服务启动命令                                                 |
| model_path           | 模型路径                                                     |
| cache                | 缓存配置:用于配置是否启用KV缓存机制                          |
| attention_backend    | 配置后端注意力                                               |
| enable_parallel      | 配置分布式并行策略                                           |
| mtp                  | MTP（Massive Text Prediction）加速推理配置                   |
| cuda_graph           | 用于捕获 GPU 操作序列并重复执行，减少内核启动开销            |
| mem_fraction_static  | 静态内存分配比例                                             |
| chunked_prefill_size | 分块预填充大小（单位：token）                                |
| dtype                | 数据类型：常见选项：float32（单精度，精度最高但内存占用大）、float16/bfloat16（半精度，内存占用减半，精度损失可控）、float8（低精度，适合高吞吐量场景） |
| torch_compile        | PyTorch 的编译优化                                           |
| embedding_tp_size    | 嵌入层（embedding layer）的张量并行（Tensor Parallel）尺寸：例如 embedding_tp_size=2 表示将嵌入层权重拆分到 2 张 GPU 上，每张 GPU 存储部分嵌入表，计算时通过跨卡通信协同完成 |
| quantization         | 量化配置：指定模型参数的量化方式（将高精度权重转换为低精度，如 int8、int4），以减少内存占用并加速推理。 |

注意：上述参数字段如有不使用的，需要填写值为空字符串或删掉，否则任务会直接退出

**所有内置的模型json文件都是当前版本的最佳性能参数，只需要修改模型路径和机器等信息即可**

## 2.3 Benchmark 测试 (benchmark.json)

| 参数             | 说明                        |
| ---------------- | --------------------------- |
| input_output_len | 输入token长度/输出token长度 |
| num_prompt       | 并发请求数                  |

```json
{
    "tasks": [
        {
            "task_name": "sglang_bench",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path models/DeepSeek-R1-W8A8-0528/vllm_quant_model"],
                "cuda_graph": [""],
                "torch_compile": [""],
                "dtype": [""],
                "cache": ["--disable-radix-cache --disable-chunked-prefix-cache"],
                "quantization": [""],
                "chunked_prefill_size": [""],
                "attention_backend": ["--attention-backend flashinfer"],
                "enable_parallel": ["--tp 16 --dp 4DeepSeek-R1-W8A8-0528"],
                "mem_fraction_static": [""],
                "embedding_tp_size": [""],
                "mtp": [" --speculative-algorithm NEXTN --speculative-draft-model-path /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8",""
                        ]
            },
            "benchmark": {
                "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json "
                "input_output_len": ["3072/1024"],
                "num_prompt": ["1", "16", "128", "256", "32", "64"]
            }
        }
    ]
}

```

其中model path和 benchmark的 ShareGPT_V3_unfiltered_cleaned_split.json需要修改为镜像内可访问的路径


## 2.2 精度测试 (acc.json)

目前只支持mmlu和ceval精度测试

- **mmlu**

| 参数     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| nsub     | 学科数，默认60                                               |
| data_dir | 如果使用mmlu数据集进行精度测试，需要准备data数据，请从https://people.eecs.berkeley.edu/~hendrycks/data.tar下载、解压, 使用此路径。此外，如果是离线环境还需要从https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken下载"cl100k_base.tiktoken"文件，放到容器内任意路径下，并且将cl100k_base.tiktoken文件重命名为9b5ad71b2ce5302211f9c61530b329a4922fc6a4(注：此目录名称为tiktoken下载的http链接的hash，如果后续下载链接有变更，则调整此目录名)，同时需要在任务配置信息中增加环境变量 TIKTOKEN_CACHE_DIR 设置为上述任意路径的绝对全路径，如下：
```json
    ......
    "environment": {
        "TIKTOKEN_CACHE_DIR": "{下载cl100k_base.tiktoken的所在的路径的绝对全路径}"
    },
    ......
```


- **ceval**

| 参数           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| random\[可选\] | 随机抽样设置，由下面两个参数构成。<br>random_seed 随机种子\[必选\]<br>random_num 抽样数\[必选\]<br>如：--random_seed 0 --random_num 50<br>如果不设置此项，则为全量测试。 |
| test_jsonl     | /workspace/ModelZoo.LLM.Inference/dataset下提供了ceval_val_cmcc.jsonl |
| timeout        | 超时设置，默认1200s |

```json
{
    "tasks": [
        {
            "task_name": "test_mmlu",
            "acc_type": "mmlu",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-W8A8-0528/vllm_quant_model"],
                "cache": ["--disable-radix-cache"],
                "attention_backend": ["--attention-backend flashinfer"],
                "enable_parallel": ["--tp 16 --dp 4 --enable-dp-attention"],
                "mtp": [
                    "--speculative-algo NEXTN --speculative-draft /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8",
                    ""
                ],
                "cuda_graph":[
                ]
            },
            "benchmark": {
                "command_base": "python3 bench_sglang.py --data_dir /model/acc/mmlu/data --nsub 60"
            }
        },
        {
            "task_name": "test_ceval",
            "acc_type": "ceval",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-W8A8-0528/vllm_quant_model"],
                "cache": ["--disable-radix-cache"],
                "attention_backend": ["--attention-backend flashinfer"],
                "enable_parallel": ["--tp 16 --dp 4 --enable-dp-attention"],
                "mtp": [
                    "--speculative-algo NEXTN --speculative-draft /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8",
                    ""
                ],
                "cuda_graph":[
                    "--cuda-graph-max-bs 64 --chunked-prefill-size 2048",
                    ""
                ]
            },
            "benchmark": {
                "command_base": "python3 run_ceval_client.py --model /models/DeepSeek-R1-W8A8-0528/vllm_quant_model  --test_jsonl /workspace/ModelZoo.LLM.Inference/dataset/ceval_val_cmcc.jsonl --batch_size 64 --random_seed 0 --random_num 50",
            }
        }
    ]
}
```

其中model path和 精度测试的data路径 需要修改为镜像内可访问的路径

# 3 输出结构

\--output-path 指定的输出目录，建议使用镜像挂载的外部目录，否则默认生成在镜像内部，当镜像移除后无法查看

输出目录大概分为三个**层级**

**层级1**: 包含多轮测试结果的总目录，每轮测试一个目录，以{时间戳}方式命名，如20250805_181420。

```markdown
└── {输出目录}/                                # --output-path
　　├── {时间戳}                               # 本轮测试的目录
　　└── total_real_progress_file.json          # 全局进度控制文件
```

**层级2**: 每轮测试目录包含该轮测试所指定的各个项目，如DeepSeek-R1-BF16-W8A8，Qwen3_32B等。

```markdown
└── {输出目录}/                                   # --output-path
　　├── {时间戳}                                # 本轮测试的目录
　　│    ├──  {模型名}                          # 本轮测试的模型名称1
　　│    └──  {模型名}                          # 本轮测试的模型名称2
　　│    └──  benchmark.csv                    # 本轮测试的精度测试结果汇总（若有）
　　│    └──  acc.csv                          # 本轮测试的benchmark测试结果汇总（若有）
　　│    └──  bench_record.log                 # 本轮测试的测试脚本执行记录和结果汇总日志
　　└── total_real_progress_file.json          # 全局进度控制文件
```

**层级3**: 每轮测试目录包含该轮测试所指定的各大类任务，如acc(精度)，benchmark(标准benchmark)等。

```markdown
{模型名}/
├── acc/                                     # 精度
│   └── real_progress_file.json              # 测试进度控制文件
└── benchmark/                               # 标准benchmark
    └── real_progress_file.json
```

**层级4**: 每种任务各自目录，包含logs(实时日志)，result(任务结果相关)。result下目录结构根据任务类型不同有所差异，具体参考下面结构。

```plaintext
acc/
　├── logs/
　│　 └── {任务名}_server{任务编号}_node{节点}.log    # 实时日志
　└── result/
　　　└── ceval|mmlu/                                # 精度任务子类型
　　　 　　└── {任务名}-server{任务编号}/              # 单次精度结果
　　　 　　　　├── *_result.json                      # 结果
　　　 　　　　├── *_result.txt                       # 过程输出
　　　 　　　　└── *_result.csv                       # 指标提取
　　　 　　　　└── *.txt                              # ceval测试会多出的一个结果文件

benchmark
　├── logs/
　│　 └── {任务名}_server{任务编号}_node{节点}.log    # 实时日志
　└── result/
　　　└── {任务名}-server{任务编号}/                   # 单任务结果
　　　 　　├── *_result.jsonl                         # 结果
　　　 　　├── *_result.txt                           # 过程输出
　　　 　　└── *_result.csv                           # 指标提取
```

# 4 启动测试

## 4.1 镜像准备

使用`docker pull ${image_name}:${tag}` 命令把modelzoo镜像拉取到本地，如果涉及多个节点的，需要在所有节点机器上拉取同一个镜像，确保多机环境一致， 多机测试时必须使用同一个镜像搭建容器， 搭建容器命令可参考如下, 其中`--security-opt seccomp=unconfined`必须加上， 否则会出现线程权限不足报错。-v 处的目录映射为推荐方式，可以根据实际情况调整

```shell
docker run -it --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband --privileged=true --group-add video --name sglang_bench --device=/dev/mem --network=host --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size '100gb' --ulimit memlock=-1 -v /data/models:/models  $image_id /bin/bash
```

## 4.2 配置机器信息

进入主节点容器内部，modelzoo目录就在 /workspace/ModelZoo.LLM.Inference

修改  /workspace/ModelZoo.LLM.Inference/models/mechines.json 使用的节点信息，填写指南参考 第2章节的 测试类型的服务器信息部分，此处DeepSeek-R1-BF16-W8A8使用的双机，只需要添加两个节点信息：

```json
{
    "machine_info": [
        {
            "ip": "192.168.0.1",
            "ifname":"****",
            "ib_hcas": "****"
        },
        {
            "ip": "192.168.0.2",
            "ifname":"****",
            "ib_hcas": "****"
        }
    ],
    "common_envs": {
        ......
  }
}
```
需要注意的是，所有任务执行时最大需使用几台机器，配置几台机器即可，否则，多余配置的机器也需要作为从节点启动。

## 4.3 配置任务信息

进入主节点容器内部，modelzoo目录就在 /workspace/ModelZoo.LLM.Inference

以DeepSeek-R1-BF16-W8A8为例，测试的任务是benchmark，那么修改 

 /workspace/ModelZoo.LLM.Inference/models/DeepSeek-R1-BF16-W8A8/benchmark.json

```json
{
    "tasks": [
        {
            "task_name": "test",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-W8A8-0528/vllm_quant_model"],
                "cuda_graph": [""],
                "torch_compile": [""],
                "dtype": [""],
                "cache": ["--disable-radix-cache --disable-chunked-prefix-cache"],
                "quantization": [""],
                "chunked_prefill_size": [""],
                "attention_backend": ["--attention-backend flashinfer"],
                "enable_parallel": ["--tp 16 --dp 4 --enable-dp-attention"],
                "mem_fraction_static": [""],
                "embedding_tp_size": [""],
                "mtp": [" --speculative-algorithm NEXTN --speculative-draft-model-path /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8"]
            },
            "benchmark": {
                "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json ",
                "input_output_len": ["3072/1024"],
                "num_prompt": ["1", "16", "32", "64", "128"]
            }
        }
    ]
}
```

配置里面默认带的是本版本最佳性能参数，需要用户手动手改model_path，mtp model_path，ShareGPT_V3_unfiltered_cleaned_split.json（如有使用）的路径为镜像内部可以访问的路径，

如果想测试其他参数组合，可以在launch_server里面加，会自动组合生成测试结果，且json支持添加多个任务

## 4.4 启动benchmark

```python
# (容器内)进入code目录
cd /workspace/ModelZoo.LLM.Inference/code
# 对mechines.json文件中除主节点以外的“所有”从节点执行（无论从节点在此任务中有没有使用到），以上面4.2中的配置信息为例，需要对ip为192.168.0.2的设备执行即可。 port 可自定义（保持主从一致），
python3 -m src.slave --local-ip 192.168.1.10 --port 20005 
# 必选参数：
--local-ip：从节点ip，与mechines.json中配置的从节点ip保持一致
# 可选参数：
--port：从节点监听的端口后，必须和主节点一致，默认是20000

# 在主节点执行， port 可自定义（保持主从一致），可不配置，默认20000
python3 -m src.master --output-path ../outputs/ --tasks ../models/DeepSeek-R1-BF16-W8A8/benchmark.json ../models/DeepSeek-R1-BF16-W8A8/acc.json --machine-config ../models/mechines.json --port 20005

# 参数说明：
--machine-config：本次测试需要的机器信息
--output-path：结果输出的根目录，最好是外部挂载进容器的目录，防止容器删了结果丢失
--tasks：指定测试的配置文件，可以指定多个配置
--port：socket的端口号，默认20000

```

测试完成日志和结果都存放在output-path，结构说明见第3章 
