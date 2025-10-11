# ModelZoo SGLang how-to-run

# 1 目录结构

```markdown
ModelZoo.LLM.Inference
├──code/
│   ├── src/
│   │   ├── connection.py
│   │   ├── master.py
│   │   ├── output.py
│   │   ├── slave.py
│   │   ├── task.py
│   ├── tools/
│   │   └── start_docker.py
│   ├──utils/
│   │   └── utils.py
│   ├── accuracy_test.sh
│   ├── bench_sglang.py
│   ├── bench_test.py
│   ├── dailytest.txt
│   ├── openai_chatcompletion_client.py
│   ├── openai_completion_client.py
│   ├── run_bench_test_batched.sh
│   ├── run_bench_test_once.sh
│   ├── run_ceval_client.py
│   ├── run_ceval_test.sh
│   ├──pd_scripts/
│   │   ├── generate_config.py
│   │   ├── launch_servers.sh
│   │   ├── template_decoder.sh
│   │   ├── template_docker.sh
│   │   ├── template_prefill.sh
├── models/
│   ├── 模型名称
│   │   ├── benchmark.json
│   │   ├── acc.json
│   │   ├── rampup_bench.json
│   │   └── search.json
│   ├── mechines.json
└── README.md
```

code下存放的是测试代码和脚本, models目录下存放的是支持的模型的配置文件

# 2 测试类型

| 测试类型      | 说明 |
| ---          | ---  |
| benchmark    | 可以组合各种参数的benchmark测试 |
| rampup_bench | 爬坡测试 |
| search       | 摸高测试，指定指标如tpot、input、output等，搜索出满足指标的最大batch |
| acc          | 精度测试，目前支持 mmlu和ceval |

服务器信息参数如下：

- **machine_info：节点配置**

| 参数     | 说明 |
| ---     | ---  |
| ip      | IP地址 |
| ifname  | 网络接口名，填写与ip匹配的网卡的名。可以参考下面命令来获取<br>ip -o addr show \| grep  "inet {IP地址}" \| awk '{print $2}' \| sed 's/://'<br>如：<br>ip -o addr show \| grep  "inet 127.0.0.1" \| awk '{print $2}' \| sed 's/://' |
| ib_hcas | 主机通道适配器<br>使用ibstat命令来获取，ibstat 输出类似如下结果<pre style="font-size: 12px; line-height: 1.2;">root@****:~# ibstat<br>CA 'mlx5_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                Base lid: 0<br>                ...<br>CA 'mlx5_1'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                ...<br>CA 'mlx5_bond_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 100<br>                ...<br>...</pre>请收集所有名称符合 mlx5_\[数字\]，并且State为Active的结果。<br>如上面，则应设置为 "ib_hcas":"mlx5_0,mlx5_1"， bond_0不需要带上。 |

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
  ]
}
```

- **tasks: 测试任务信息**

| 参数         | 说明 |
| ---         | --- |
| task_name   | 任务名称 |
| launch_mode | 模式：online、offline |
| world_size  | 跑本任务测试需要用到的卡数量 |
| server_port | 端口号 |

- **environment：环境变量配置**

| 参数                                              | 说明 |
| ---                                               | --- |
| MACA_SMALL_PAGESIZE_ENABLE                        | 页面大小优化 |
| TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP | Triton 编译器优化 |
| TRITON_ENABLE_MACA_CHAIN_DOT_OPT                  | Triton 编译器的链式 Dot 操作优化 |
| PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM            | PyTorch 的优先级流（Priority Stream）优化 |
| CUDA_GRAPH_DP_USE_SUM_BS                          | CUDA Graph 的动态批处理优化，默认False |

- **launch_server：服务配置**

| 参数                  | 说明 |
| ---                  | --- |
| command_base         | 服务启动命令 |
| model_path           | 模型路径 |
| cache                | 缓存配置:用于配置是否启用KV缓存机制 |
| attention_backend    | 配置后端注意力 |
| enable_parallel      | 配置分布式并行策略 |
| mtp                  | MTP（Massive Text Prediction）加速推理配置 |
| cuda_graph           | 用于捕获 GPU 操作序列并重复执行，减少内核启动开销 |
| mem_fraction_static  | 静态内存分配比例 |
| chunked_prefill_size | 分块预填充大小（单位：token） |
| dtype                | 数据类型：常见选项：float32（单精度，精度最高但内存占用大）、float16/bfloat16（半精度，内存占用减半，精度损失可控）、float8（低精度，适合高吞吐量场景） |
| torch_compile        | PyTorch 的编译优化 |
| embedding_tp_size    | 嵌入层（embedding layer）的张量并行（Tensor Parallel）尺寸：例如 embedding_tp_size=2 表示将嵌入层权重拆分到 2 张 GPU 上，每张 GPU 存储部分嵌入表，计算时通过跨卡通信协同完成 |
| quantization         | 量化配置：指定模型参数的量化方式（将高精度权重转换为低精度，如 int8、int4），以减少内存占用并加速推理。 |

**所有内置的模型json文件都是当前版本的最佳性能参数，只需要修改模型路径和机器等信息即可**

## 2.1 Benchmark 测试 (benchmark.json)

| 参数              | 说明 |
| ---              | --- |
| input_output_len | 输入token长度/输出token长度 |
| num_prompt       | 并发请求数 |

```json
{
    "tasks": [
        {
            "task_name": "sglang_bench",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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
                "mtp": [" --speculative-algorithm NEXTN --speculative-draft-model-path /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8",""
                        ]
            },
            "benchmark": {
                "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json ",
                "input_output_len": ["3072/1024", "128/1024"],
                "num_prompt": ["1", "16", "128", "256", "32", "64"]
            }
        }
    ]
}

```

其中model path和 benchmark的 ShareGPT_V3_unfiltered_cleaned_split.json需要修改为镜像内可访问的路径

## 2.2 爬坡测试 (rampup_bench.json)

| 参数                    | 说明 |
| ---                     | --- |
| max_concurrent_requests | 爬坡过程参数，只能有一个列表<br>如：\["1","4","8","16"\] |
| rampup_period           | 爬坡间隔参数，可以有多组，每组长度都必须和max_concurrent_requests等长<br>如：\["5,5,10,10"\]<br>或：\["5,5,10,10", "10,10,20,20"\] |
| requests_configs        | 请求数量控制参数。由下面三个参数构成<br>num\_warmup\_requests\_ratio：warmup倍率\[必选\]<br>num\_benchmark\_requests\_ratio：请求数倍率\[必选\]<br>least\_requests\_num：每次爬坡发送最小请求数\[可选\]<br>如：--num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16<br>或：--num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16 --least_requests_num 16 |
| input_output_len        | 输入token长度/输出token长度 |

```json
{
    "tasks": [
        {
            "task_name": "DS-R1-W8A8-0724",
            "launch_mode":"online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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
                "mtp": [
                    "--speculative-algo NEXTN --speculative-draft /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8"
                ]
            },
            "benchmark": {
                "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json ",
                "max_concurrent_requests":["1","4","8","16","32","64","80","100"],
                "rampup_period": ["5,5,10,10,20,20,30,30"],
                "requests_configs": [
                    " --num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16 "
                ], 
                "input_output_len": ["128/128"]
            }
        }
    ]
}
```

其中model path和 测试用的 ShareGPT_V3_unfiltered_cleaned_split.json需要修改为镜像内可访问的路径

## 2.3 摸高测试 (search.json)

| 参数              | 说明 |
| ---               | --- |
| input_output_len  | 输入token长度/输出token长度 |
| batch_size_config | 摸高测试，batch size 相关配置 |
| range             | batch size 范围，包含起始值 ，但不包含结束值 |
| steps             | batch size 增加步幅 |
| max_ttft          | 限定最大ttft，单位：毫秒 |
| max_tpot          | 限定最大tpot，单位：毫秒 |

```json
{
    "tasks": [
        {
            "task_name": "sglang_bench_search",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
                "cuda_graph": [""],
                "torch_compile": [""],
                "dtype": [""],
                "cache": ["--disable-radix-cache --disable-chunked-prefix-cache"],
                "quantization": [""],
                "chunked_prefill_size": [""],
                "attention_backend": ["--attention-backend flashinfer"],
                "enable_parallel": ["--tp 16 --dp 4 --enable-dp-attention"],
                "mem_fraction_static": ["--mem-fraction-static 0.85"],
                "embedding_tp_size": [""],
                "mtp": [""]
            },
            "benchmark": {
                "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json",
                "input_output_len": ["3072/1024"], // intput_len 3072 / output_len 1024
                "batch_size_config": {  // batch size 配置
                    "range": [          // batch size 范围从1开始到65不包含65
                        "1",
                        "65"
                    ],
                    "steps": "3"   // batch size 增加步幅，例如：1、4、7 ...
                },
                "max_ttft": "8000", // 限定最大ttft
                "max_tpot": "80"    // 限定最大tpot
            }
        }
    ]
}
```

其中model path 和 测试用的ShareGPT_V3_unfiltered_cleaned_split.json需要修改为镜像内可访问的路径

## 2.4 精度测试 (acc.json)

目前只支持mmlu和ceval精度测试

- **mmlu**

| 参数 | 说明 |
| ---  | --- |
| nsub | 学科数，默认60 |

- **ceval**

| 参数            | 说明 |
| ---             | --- |
| random\[可选\]  | 随机抽样设置，由下面两个参数构成。<br>random_seed 随机种子\[必选\]<br>random_num 抽样数\[必选\]<br>如：--random_seed 0 --random_num 50<br>如果不设置此项，则为全量测试。 |

```json
{
    "tasks": [
        {
            "task_name": "DS-R1-W8A8-mmlu",
            "acc_type": "mmlu",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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
                "command_base": "python3 bench_sglang.py --data_dir /model/acc/mmlu/data --nsub 60"
            }
        },
        {
            "task_name": "DS-R1-W8A8-ceval",
            "acc_type": "ceval",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM":"1",
                "CUDA_GRAPH_DP_USE_SUM_BS":"False",
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP":"1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "FUSED_RMSNORM_QUANT":"True"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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
                "command_base": "python3 run_ceval_client.py --model /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model  --test_jsonl /models/acc/ceval/ceval_val_cmcc.jsonl --batch_size 64",
                "random": ["--random_seed 0 --random_num 50"]
            }
        }
    ]
}
```

其中model path和 精度测试的data路径 需要修改为镜像内可访问的路径

# 3 输出结构

\--output-path 指定的输出目录，建议使用镜像挂载的外部目录，否则默认生成在镜像内部，当镜像移除后无法查看

输出目录大概分为三个**层级**

**层级1**: 包含多轮测试结果的总目录，每轮测试一个目录，以{时间戳}方式命名，如20250805_181420。

```markdown
└── {输出目录}/                                # --output-path
　　├── {时间戳}                               # 本轮测试的目录
　　└── total_real_progress_file.json          # 全局进度控制文件
```

**层级2**: 每轮测试目录包含该轮测试所指定的各个项目，如DeepSeek-R1-BF16-W8A8，Qwen3_32B等。

```markdown
└── outputs/                                   # --output-path
　　├── {时间戳}                                # 本轮测试的目录
　　│    ├──  {模型名}                          # 本轮测试的模型名称1
　　│    └──  {模型名}                          # 本轮测试的模型名称2
　　└── total_real_progress_file.json           # 全局进度控制文件
```

**层级3**: 每轮测试目录包含该轮测试所指定的各大类任务，如acc(精度)，benchmark(标准benchmark)等。

```markdown
{模型名}/
├── acc/                                     # 精度
│   └── real_progress_file.json              # 测试进度控制文件
├── benchmark/                               # 标准benchmark
│   └── real_progress_file.json
├── rampup/                                  # 爬坡
│   └── real_progress_file.json
└── search/                                  # 摸高
    └── real_progress_file.json
```

**层级4**: 每种任务各自目录，包含logs(实时日志)，result(任务结果相关)。logs/result下根据任务的 online/offline 类型各有一个子目录。result下目录结构根据任务类型不同有所差异，具体参考下面结构。

```plaintext
acc/
　├── logs/
　│　 └── online|offline/
　│　　　 └── {任务名}_server{任务编号}_node{节点}.log    # 实时日志
　└── result/
　　　└── online|offline/
　　　　　├── ceval|mmlu/                                # 精度任务子类型
　　　　　│　　└── {任务名}-server{任务编号}/              # 单次精度结果
　　　　　│　　　　├── *_result.jsonl                     # 结果
　　　　　│　　　　├── *{_result}.txt                     # 过程输出
　　　　　│　　　　└── *_result.csv                       # 指标提取
　　　　　└── {时间戳}_result.csv                         # 所有精度指标汇总

benchmark|perf|rampup
　├── logs/
　│　 └── online|offline/
　│　　　 └── {任务名}_server{任务编号}_node{节点}.log    # 实时日志
　└── result/
　　　└── online|offline/
　　　　　├── {任务名}-server{任务编号}/                   # 单任务结果
　　　　　│　　├── *_result.jsonl                         # 结果
　　　　　│　　├── *_result.txt                           # 过程输出
　　　　　│　　└── *_result.csv                           # 指标提取
　　　　　└── {时间戳}_result.csv                         # 各类指标汇总

search/
　├── logs/
　│　 └── online|offline/
　│　　　 └── {任务名}_server{任务编号}_node{节点}.log    # 实时日志
　└── result/
　　　└── online|offline/
　　　　　├── total_ttft_{ttft}-tpot_{tpot}/             # 所有摸高结果汇总解析
　　　　　│　　├── *.png                                 # 
　　　　　│　　└── *.csv                                 # 
　　　　　├── {任务名}-server{任务编号}/                  # 单次摸高结果
　　　　　│　　├── ttft_{ttft}-tpot_{tpot}/              # 解析
　　　　　│　　│　　├── *.png                            # 
　　　　　│　　│　　└── *.csv                            # 
　　　　　│　　├── *_result.jsonl                        # 结果
　　　　　│　　├── *_result.txt                          # 过程输出
　　　　　│　　└── *_result.csv                          # 指标提取
　　　　　├── *.png                                      # 
　　　　　└── {时间戳}_result.csv                        # 摸高指标汇总
```

# 4 启动测试

## 4.1 容器内部手动启动方式

### 4.1.1 镜像准备

使用`docker pull ${image_name}:${tag}` 命令把modelzoo镜像拉取到本地，如果涉及多个节点的，需要在所有节点机器上拉取同一个镜像，确保多机环境一致， 多机测试时必须使用同一个镜像搭建容器， 搭建容器命令可参考如下, 其中`--security-opt seccomp=unconfined`必须加上， 否则会出现线程权限不足报错。-v 处的目录映射为推荐方式，可以根据实际情况调整

```shell
docker run -it --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband --privileged=true --group-add video --name sglang_bench --device=/dev/mem --network=host --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size '100gb' --ulimit memlock=-1 -v /data/models:/models  $image_id /bin/bash
```

### 4.1.2 配置机器信息

进入主节点容器内部，modelzoo目录就在 /workspace/ModelZoo.LLM.Inference

修改  /workspace/ModelZoo.LLM.Inference/models/mechines.json 使用的节点信息，填写指南参考 第2章节的 测试类型的服务器信息部分，此处DeepSeek-R1-BF16-W8A8使用的双机，只需要添加两个节点信息：

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
}
```

### 4.1.3 配置任务信息

进入主节点容器内部，modelzoo目录就在 /workspace/ModelZoo.LLM.Inference

以DeepSeek-R1-BF16-W8A8为例，测试的任务是benchmark，那么修改 

 /workspace/ModelZoo.LLM.Inference/models/DeepSeek-R1-BF16-W8A8/benchmark.json

```json
{
    "tasks": [
        {
            "task_name": "DS-R1-W8A8",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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

配置里面默认带的是本版本最佳性能参数，需要用户手动手改model_path，mtp model_path，ShareGPT_V3_unfiltered_cleaned_split.json（如有使用）的路径为镜像内部可以访问的路径，

如果想测试其他参数组合，可以在launch_server里面加，会自动组合生成测试结果，且json支持添加多个任务

### 4.1.4 启动benchmark

```python
# (容器内)进入code目录
cd /workspace/ModelZoo.LLM.Inference/code
# 在所有从节点执行， port 可自定义（保持主从一致），
python3 -m src.slave --port 20005
# 在主节点执行， port 可自定义（保持主从一致），可不配置，默认20000
python3 -m src.master --output-path ../outputs/ --tasks ../models/DeepSeek-R1-BF16-W8A8/benchmark.json ../models/DeepSeek-R1-BF16-W8A8/acc.json --machine-config ../models/mechines.json --port 20005

# 参数说明：
--output-path：结果输出的根目录，最好是外部挂载进容器的目录，防止容器删了结果丢失
--tasks：指定测试的配置文件，可以指定多个配置
--port：socket的端口号，默认
# 增量功能
--incremental-mode：测试当前任务中非PASS的项
--specify-task：从上次中断处继续执行测试
```

测试完成日志和结果都存放在output-path，结构说明见第3章 

## 4.2 容器外部一键式启动方式

### 4.2.1 镜像和代码准备

使用`docker pull ${image_name}:${tag}` 命令把modelzoo镜像拉取到本地，如果涉及多个节点的，需要在所有节点机器上拉取同一个镜像，确保多机环境一致

在主节点上创建一个临时镜像，可以参考下面命令：

```shell
docker run -it --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband --privileged=true --group-add video --name sglang_tmp --device=/dev/mem --network=host --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size '100gb' --ulimit memlock=-1 -v /data/models:/models  $image_id /bin/bash
```

将本机路径挂载到镜像里面，方便拷贝modelzoo代码出来

进入镜像内部，将 /workspace/ModelZoo.LLM.Inference 拷贝到外面host路径(假如是/mnt/data)

然后退出镜像，确保host上/mnt/data/ModelZoo.LLM.Inference存在，同时临时镜像可以删除

后续操作都是在主节点侧镜像外部

### 2.2.2 配置机器信息

进入主节点host的/mnt/data/ModelZoo.LLM.Inference目录

修改  /mnt/data/ModelZoo.LLM.Inference/models/mechines.json 使用的节点信息，填写指南参考 第2章节的 测试类型的服务器信息部分，此处DeepSeek-R1-BF16-W8A8使用的双机，只需要添加两个节点信息：

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
}
```

### 4.2.3 配置任务信息

进入主节点host的/mnt/data/ModelZoo.LLM.Inference目录

以DeepSeek-R1-BF16-W8A8为例，测试的任务是benchmark，那么修改 

  /mnt/data/ModelZoo.LLM.Inference/models/DeepSeek-R1-BF16-W8A8/benchmark.json

```json
{
    "tasks": [
        {
            "task_name": "DS-R1-W8A8",
            "launch_mode": "online",
            "world_size": "16",
            "server_port": "5005",
            "environment": {
                "MACA_SMALL_PAGESIZE_ENABLE": "1",
                "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP": "1",
                "TRITON_ENABLE_MACA_CHAIN_DOT_OPT": "1",
                "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM": "1"
            },
            "launch_server": {
                "command_base": ["python3 -m sglang.launch_server --trust-remote-code"],
                "model_path": ["--model-path /models/DeepSeek-R1-BF16_W8A8/vllm_quant_model"],
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

配置里面默认带的是本版本最佳性能参数，需要用户手动手改model_path，mtp model_path，ShareGPT_V3_unfiltered_cleaned_split.json（如有使用）的路径为镜像内部可以访问的路径，

如果想测试其他参数组合，可以在launch_server里面加，会自动组合生成测试结果，且json支持添加多个任务

### 4.2.4 启动一键式benchmark

```shell
# 进入host /mnt/data/ModelZoo.LLM.Inference/code目录
cd /mnt/data/ModelZoo.LLM.Inference/code

python3 -m tools.start_docker --container-name sglang --container-images pub-registry1.metax-tech.com/ai-opentest/master/maca/modelzoo.llm.sglang:maca.ai20250813-124-torch2.6-py310-ubuntu22.04-amd64  --docker-v /models:/models /mnt/data:/mnt/data --rm-exist-docker --machine-config ../models/mechines.json --tasks-config ../models/DeepSeek-R1-BF16-W8A8/benchmark.json ../models/DeepSeek-R1-BF16-W8A8/acc.json  --output-path /mnt/data/benchmark_outputs
```

需要注意的是，任务完成会自动停止镜像，建议output-path设置镜像外部host主机挂载进去的目录，免得镜像被误删后无法查看任务结果

测试完成日志和结果都存放在output-path，结构说明见第3章 

参数说明：

\--container-name：容器名称

\--container-images：镜像，可以指定多个

\--container-cycles：每种镜像执行的次数

\--pull-images：不加此参数代表默认镜像已经存在

\--rm-exist-docker：是否删除旧容器创建新的

\--docker-v：挂载目录，尤其是模型目录、数据集目录

\--prepare-docker-cmds：指定docker启动成功后的初始化操作

\--target-path:容器中ModelZoo项目路径，不指定使用默认即可

\--machine-config：设备配置文件

\--output-path：结果输出的路径

\--tasks-config：指定benchmark的配置文件，可以指定多个

\--incremental-mode：测试当前任务中非PASS的项

\--specify-task：从上次中断处继续执行测试

\--user：服务器的用户名

\--port：socket的端口号

# 5 pd分离服务启动
使用code/pd_scripts下脚本可实现一键启动sglang pd分离场景下的server端，包括prefill节点、deocde节点以及mini_lb服务，下面对使用步骤进行说明

## 5.1 前提条件
脚本使用依赖下述条件

1. 执行脚本的节点能ssh免密连接到其它节点
2. 通过配置/etc/hosts，保证每个节点使用`hostname -i`命令获取到的ip与ssh连接该节点的ip一致

## 5.2 配置修改

### 5.2.1 generate_config.py修改
generate_config主要用于生成多机启动的环境变量配置，根据需求进行修改

```python
container_name='sglang-benchmark-server' # 测试启动容器名字
model_path='/oschina0/kychina/models/DeepSeek-R1-BF16_W8A8/vllm_quant_model/' # 模型路径
port=8000 # 在线测试发送请求的端口

# 节点ip列表
host_list = [
    "192.168.12.15", # decode node 0 
    "192.168.12.17", # decode node 1

    "192.168.12.11", # prefill node 0
    "192.168.12.13", # prefill node 1
]

num_prefill = 1 # prefill实例数量
num_decoder = 1 # decode实例数量
num_nodes_per_prefill = 2 # 每个prefill实例使用节点数量
num_nodes_per_decoder = 2 # 每个decode实例使用节点数量
prefill_pp_size = 2 # prefill使用pp并行的pp_size
```

### 5.2.2 template_docker.sh修改
template_docker.sh中是docker启动容器的命令，根据需求修改镜像、挂载目录

```shell
set -x

WORKDIR=$(dirname "$(readlink -f "$0")")
cd $WORKDIR

source configs/$(hostname -i).env

docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

DOCKER_IMAGE=pub-registry1.metax-tech.com/ai-opentest/master/maca/sglang:0.5.1-maca.ai20251011-38-torch2.6-py310-ubuntu22.04-amd64 # 镜像

if [[ $ROLE == "decoder" ]]; then
    LAUNCH_SCRIPT=template_decoder.sh
else
    LAUNCH_SCRIPT=template_prefill.sh
fi

docker run -itd --rm --name=$CONTAINER_NAME \
            --net=host \
            --uts=host \
            --ipc=host \
            --device=/dev/dri \
            --device=/dev/mxcd  \
            --device=/dev/infiniband \
            --privileged=true \
            --group-add video \
            --security-opt seccomp=unconfined \
            --security-opt apparmor=unconfined \
            --shm-size 100gb \
            --ulimit memlock=-1 \
            -d \
            -v /data/:/data/ \
            -v /oschina0/kychina/:/oschina0/kychina/ \ # 挂载目录，自行增减
            -v $WORKDIR:/workspace \ # 不要修改这个
            --workdir=/workspace \
            --runtime=runc -t $DOCKER_IMAGE /bin/bash -c "source /etc/profile && source ~/.bashrc && bash $LAUNCH_SCRIPT"
```

### 5.2.3 template_prefill.sh修改
template_prefill.sh中是prefill节点启动命令，多机相关参数无须修改，sglang使用的环境变量、切分方式、内存设置等参数根据需求进行修改

```shell
#!/bin/bash

set -x
pidof python | xargs -n 1 kill -9
source configs/$(hostname -i).env


export MCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export GLOO_SOCKET_IFNAME=inbond1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MACA_DIRECT_DISPATCH=1
export MACA_GRAPH_LAUNCH_QUEUE_POLICY=3
export MCDBG_GRAPH_LAUNCH_QUEUE_POLICY=3
# export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
# export MACA_QUEUE_SCHEDULE_POLICY=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export CUDA_GRAPH_DP_USE_SUM_BS=0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
# export SGLANG_PP_LAYER_PARTITION="32,29"
export SGLANG_TORCH_PROFILER_DIR=/oschina0/kychina/yunlei/tmp_file/

PP=$PREFILL_PP_SIZE
TP=$(( $NNODES * 8 / $PP ))
MEMORY_FRACTION=0.84

env
# --dist-init-addr --nnodes --node-rank多机参数自动生成，不用修改
python3 -m sglang.launch_server \
    --model-path $MODEL \ # model_path在generate_config.py中进行指定，此处通过环境变量获取，也可直接指定
    --dist-init-addr $DIST_INIT_ADDR  \
    --host $(hostname -i) \
    --port 30000 \
    --tp $TP \
    --pp $PP \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --disable-radix-cache  \
    --attention-backend flashinfer  \
    --mem-fraction-static ${MEMORY_FRACTION} \
    --disaggregation-mode prefill  \
    --disaggregation-ib-device  $MCCL_IB_HCA \
    --trust-remote-code 2>&1 | tee logs/prefill_$(hostname -i).log
```
### 5.2.4 template_decoder.sh修改
template_decoder.sh中是decode节点启动命令，多机相关参数无须修改，sglang使用的环境变量、切分方式、内存设置等参数根据需求进行修改
```shell
#!/bin/bash

set -x
pidof python | xargs -n 1 kill -9
source configs/$(hostname -i).env

export MCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export GLOO_SOCKET_IFNAME=inbond1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MACA_DIRECT_DISPATCH=1
export MACA_GRAPH_LAUNCH_QUEUE_POLICY=3
export MCDBG_GRAPH_LAUNCH_QUEUE_POLICY=3
# export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
# export MACA_QUEUE_SCHEDULE_POLICY=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export CUDA_GRAPH_DP_USE_SUM_BS=0
export SGLANG_TORCH_PROFILER_DIR=/oschina0/kychina/yunlei/tmp_file/


TP=16
DP=4
MEMORY_FRACTION=0.86

env

# --dist-init-addr --nnodes --node-rank多机参数自动生成，不用修改
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --dist-init-addr $DIST_INIT_ADDR  \
    --host $(hostname -i) \
    --port 30001 \
    --tp $TP \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --disable-radix-cache  \
    --attention-backend flashinfer  \
    --quantization w8a8_int8 \
    --dp $DP \
    --enable-dp-attention \
    --mem-fraction-static ${MEMORY_FRACTION} \
    --disaggregation-mode decode  \
    --disaggregation-ib-device  $MCCL_IB_HCA \
    --trust-remote-code 2>&1 | tee logs/decoder_$(hostname -i).log

```

### 5.2.5 启动命令
修改配置文件之后，在code/pd_scripts中执行launch_servers.sh即可启动
```shell
bash launch_servers.sh
```

### 5.2.6 输出说明
启动之后会在code/pd_scripts下生成logs和configs目录
```markdown
│   ├──pd_scripts/
│   │   ├── logs 存放日志文件
│   │   │   ├── prefill_192.168.12.11.log
│   │   │   ├── prefill_192.168.12.13.log
│   │   │   ├── decoder_192.168.12.15.log
│   │   │   ├── decoder_192.168.12.17.log
│   │   │   ├── mini_lb.log
│   │   ├── configs 脚本运行生成的中间配置文件，可以不关注
```