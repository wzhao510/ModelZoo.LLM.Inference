# ModelZoo SGLang ReadMe

# 1 目录结构

```markdown
sgl
├──code/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── benchmark.py
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
│   ├──pd_scripts/
│   │   ├── generate_config.py
│   │   ├── launch_servers.sh
│   │   ├── template_decoder.sh
│   │   ├── template_docker.sh
│   │   ├── template_prefill.sh
├── dataset/
│   └── ceval_val_cmcc.jsonl
├── models/
│   ├── 模型名称
│   │   ├── config.json
│   ├── mechines.json
└── README.md
```

code下存放的是测试代码和脚本, models目录下存放的是支持的模型的配置文件

# 2 测试类型

| 测试类型     | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| perf         | 可以组合各种参数的性能测试                              |
| acc          | 精度测试，目前支持 mmlu和ceval                               |

## 2.1 服务器信息配置 (mechines.json)

包含节点信息和通用环境变量配置

- **machine_info：节点配置**

| 参数    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| ip      | IP地址                                                       |
| ifname  | 网络接口名，填写与ip匹配的网卡的名。可以在host机器上使用命令ifconfig，然后查看ip对应的网卡名称|
| ib_hcas | 主机通道适配器<br>在host机器上使用ibstat命令来获取，ibstat 输出类似如下结果<pre style="font-size: 12px; line-height: 1.2;">root@****:~# ibstat<br>CA 'mlx5_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                Base lid: 0<br>                ...<br>CA 'mlx5_1'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 200<br>                ...<br>CA 'mlx5_bond_0'<br>        CA type: ****<br>        Number of ports: 1<br>        ...<br>        Port 1:<br>                State: Active<br>                Physical state: LinkUp<br>                Rate: 100<br>                ...<br>...</pre>请收集所有名称符合 mlx5_\[数字\]，并且State为Active的结果。<br>如上面，则应设置为 "ib_hcas":"mlx5_0,mlx5_1"， bond_0不需要带上。 |

- **environments：公共环境变量配置**

| 参数                                              | 说明                                            |
| ------------------------------------------------- | -----------------------------------------------|
| MACA_SMALL_PAGESIZE_ENABLE                        | 页面大小优化                                    |
| TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP | Triton 编译器优化                               |
| TRITON_ENABLE_MACA_CHAIN_DOT_OPT                  | Triton 编译器的链式 Dot 操作优化                 |
| MACA_DIRECT_DISPATCH                              | 开启 direct dispatch 功能                       |
| MCDBG_GRAPH_LAUNCH_QUEUE_POLICY                   | 设置 graph 内部创建的 stream/queue 的优先级为high |
| MACA_GRAPH_LAUNCH_QUEUE_POLICY                    | 设置 graph 内部创建的 stream/queue 的优先级为high |
| PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM            | PyTorch 的优先级流（Priority Stream）优化         |
| MACA_QUEUE_SCHEDULE_POLICY                        | MACA 队列调度策略设置                             |

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
  "environments": {
    "default_envs": [
      "MACA_SMALL_PAGESIZE_ENABLE=1",
      "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1",
      "TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1",
      "PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1",
      "MACA_QUEUE_SCHEDULE_POLICY=1"
    ],
    "specific_scenario_envs": [
      "MACA_SMALL_PAGESIZE_ENABLE=1",
      "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1",
      "TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1",
      "MACA_DIRECT_DISPATCH=1",
      "MCDBG_GRAPH_LAUNCH_QUEUE_POLICY=3",
      "MACA_GRAPH_LAUNCH_QUEUE_POLICY=3"
    ],
    "mmlu_envs" : [
        "TIKTOKEN_CACHE_DIR=/models/"
    ]
  }
}
```
这里可以配置多组环境变量，以组名区分，如default_envs、specific_scenario_envs，不同的任务可以通过在task和benchmark配置中的**environment**字段添加组名（如specific_scenario_envs）来直接引用对应的环境变量。

注意：**default_envs**是针对当前版本提供的默认服务启动参数的最优环境变量；如果变更服务启动参数，比如由TP切分改为DP切分，当前的默认的环境变量可能不是最优，可以尝试使用**specific_scenario_envs**，当前版本测试发现对于DeepSeek TP并行和 Qwen3 235B PP并行，**specific_scenario_envs**环境变量是最优的

- **replacements：公共变量替换**

鉴于模型路径、精度测试数据路径和测试集路径在不同的机器上路径不同，故在sgl/models/mechines.json中提供公共变量替换功能，如下：

```json
{
    ......
    "replacements": {
        "random-dataset-path": "/models/ShareGPT_V3_unfiltered_cleaned_split.json",
        "mmlu-data-path": "/models/acc/mmlu/data",
        "ceval-data-path": "/workspace/ModelZoo.LLM.Inference/dataset/ceval_val_cmcc.jsonl",

        "DeepSeek-R1-W8A8-model-path": "/models/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model",
        "DeepSeek-R1-W8A8-draft-model-path": "/models/DeepSeek-R1-NextN-Channel-INT8"
        ......
  }
    ......
}
```
在config.json中只需要通过 **${xx}** 引用即可，运行时会自动替换此变量：

```json
{
    "server_cmds": {
        "server_cmd": [
            ["python3 -m sglang.launch_server --trust-remote-code"],
            ["--model-path ${DeepSeek-R1-W8A8-model-path}"],
            ......
        ]
    }
    ......
}
```
替换后：
```json
{
    "server_cmds": {
        "server_cmd": [
            ["python3 -m sglang.launch_server --trust-remote-code"],
            ["--model-path /models/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model"],
            ......
        ]
    }
    ......
}
```

注意这个变量替换只会替换task指定的config.json中的变量，不会替换mechines.json本身的变量


## 2.2 测试通用配置说明

- **通用server命令**

在config.json的最前面可以配置一些通用的server命令，然后各个task中的**launch_server**参数可以通过命令的名称来引用，如下：
```json
{
    "server_cmds": {
        "server_cmd": [
            ["python3 -m sglang.launch_server --trust-remote-code"],
            ["--model-path /mnt/shared_data/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model"],
            ["--disable-radix-cache"],
            ["--attention-backend flashinfer"],
            ["--tp 16 --dp 8 --enable-dp-attention --enable-dp-lm-head"],
            [" --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/shared_data/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8"],
            ["--disable-shared-experts-fusion"]
        ]
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "launch_server": "server_cmd",
            ......
        }
    },
    ......
}
```
如上在task中可以直接通过引用已经定义的server命令名称来设置当前task的启动server命令，如果需要引用多个，则以分号分隔（如"server_cmd;server_cmd2"）；当然task中的**launch_server**也可以不引用，直接和上面的**server_cmd**一样用list来设置自己的命令。


- **通用benchmark命令**

在config.json的最前面也可以配置一些通用的benchmark命令，然后各个task中的**benchmark**参数可以通过命令的名称来引用，如下：
```json
{
    ......
    "benchmark_cmds": {
        "random": {
            "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json ",
            "input_output_len": ["3072/1024"],
            "max_concurrency": ["1", "16", "32", "64", "128", "256"],
            "num_prompt_times": 5
        },
        "ceval": {
            "type": "ceval",
            "command_base": "python3  run_ceval_client.py --model /models/DeepSeek-R1-0528-BF16-W8A8  --test_jsonl /workspace/ModelZoo.LLM.Inference/dataset/ceval_val_cmcc.jsonl --batch_size 64 --random_seed 0 --random_num 5"
        },
        "mmlu": {
            "type": "mmlu",
            "environment": ["mmlu_envs"],
            "command_base": "python3 bench_sglang.py --data_dir /models/acc/mmlu/data --nsub 60"
        }
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "benchmark": "random;ceval;mmlu"
            ......
        }
    }
    ......
}
```
如上在task中可以直接通过引用已经定义的benchmark命令名称来设置当前task的启动benchmark命令，如果需要引用多个，则以分号分隔（如"random;ceval"）；当然task中的**benchmark**也可以不引用，直接和上面的**benchmark_cmds**中的命令一样用结构来设置自己的命令。


- **tasks: 测试任务信息**

| 参数        | 说明                                  |
| ----------- | -----------------------------------  |
| launch_mode | 模式：online、offline，默认 online     |
| server_port | http server 端口号，默认30000          |
| dist_port   | 卡间通信组的端口号，默认5000            |

- **environment：任务独有环境变量配置**

| 参数                                              | 说明                                      |
| ------------------------------------------------- | ----------------------------------------- |

如果不设置或者为空，则此任务使用**mechines.json**中的**default_envs**环境变量配置
可以添加独有环境变量；也可以直接添加**mechines.json**中的环境变量组名来直接引用已有的环境变量，如：
```json
    ......
    "environment": [
        "specific_scenario_envs"
    ],
    ......
```
上述这个任务配置就会直接使用**mechines.json**中的**specific_scenario_envs**的环境变量配置。


**所有内置的模型config.json文件都是当前版本的最佳性能参数，只需要修改模型路径和机器等信息即可**

## 2.3 Perf 测试 (config.json)

| 参数             | 说明                                              |
| ---------------- | ------------------------------------------------- |
| type             | 测试类型，支持perf，ceval，mmlu，默认为perf         |
| input_output_len | 输入token长度/输出token长度                        |
| num_prompt       | 并发请求数，为列表                                         |
| max_concurrency  | 最大并发数，为列表，如果不设置，则并发数和num_prompt相等，必须和num_prompt_times一起使用，且不需要配置num_prompt参数|
| num_prompt_times | 发送的请求数是最大并发数的多少倍，如果是单个数字，则会以max_concurrency里面的所有并发数乘上这个倍数作为num_prompts，如果是个列表，则会将这个列表与max_concurrency列表对应，只有存在对应关系的max_concurrency会乘上这个列表的值作为num_prompts|

- 传统方式，不带max_concurrency参数的如下：
```json
{
    ......
    "benchmark_cmds": {
        "random": {
            "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json",
            "input_output_len": ["3072/1024"],
            "num_prompt": ["1", "16", "32", "64", "128"]
        }
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "benchmark": "random"
            ......
        }
    }
    ......
}

```
- max_concurrency方式，且num_prompt_times为单个数字：
```json
{
    ......
    "benchmark_cmds": {
        "random": {
            "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json",
            "input_output_len": ["3072/1024"],
            "max_concurrency": ["1", "16", "32", "64", "128"],
            "num_prompt_times": 5
        }
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "benchmark": "random"
            ......
        }
    }
    ......
}
```
这是对所有并发都跑5倍prompts的测试配置，比如bs=1组成的测试命令是 **--random-input-len 3072 --random-output-len 1024 --num-prompts 5 --max-concurrency 1**

- max_concurrency方式，且num_prompt_times为列表，这种方式可对不同并发设置不同倍数的prompts：
```json
{
    ......
    "benchmark_cmds": {
        "random": {
            "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json",
            "input_output_len": ["3072/1024"],
            "max_concurrency": ["1", "16", "32", "64", "128"],
            "num_prompt_times": [2, 3]
        }
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "benchmark": "random"
            ......
        }
    }
    ......
}
```
如上，对于bs=1则跑2倍并发：**--random-input-len 3072 --random-output-len 1024 --num-prompts 2 --max-concurrency 1**

对于bs=16,则跑3倍并发：**--random-input-len 3072 --random-output-len 1024 --num-prompts 48 --max-concurrency 16**

对于其他的bs，没有配置prompts倍数，则不添加prompts和max_concurrency相等，比如bs=32：**--random-input-len 3072 --random-output-len 1024 --num-prompts 32 --max-concurrency 32**

其中model path和 benchmark的 ShareGPT_V3_unfiltered_cleaned_split.json需要修改为镜像内可访问的路径


## 2.2 精度测试 (config.json)

目前只支持mmlu和ceval精度测试

- **mmlu**

| 参数     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| nsub     | 学科数，默认60                                               |
| data_dir | 如果使用mmlu数据集进行精度测试，需要准备data数据，请从https://people.eecs.berkeley.edu/~hendrycks/data.tar下载、解压, 使用此路径。此外，如果是离线环境还需要从https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken下载"cl100k_base.tiktoken"文件，放到容器内任意路径下，并且将cl100k_base.tiktoken文件重命名为9b5ad71b2ce5302211f9c61530b329a4922fc6a4(注：此目录名称为tiktoken下载的http链接的hash，如果后续下载链接有变更，则调整此目录名)，然后直接将sgl/models/mechines.json中的 TIKTOKEN_CACHE_DIR 设置为上述任意路径的绝对全路径既可以，如下：
```json
    "environments": {
      ......
      "mmlu_envs" : [
          "TIKTOKEN_CACHE_DIR={下载cl100k_base.tiktoken的所在的路径的绝对全路径}"
      ]
    },
```
或者在任务的benchmark配置信息中增加环境变量：
```json
    ......
    "environment": [
        "TIKTOKEN_CACHE_DIR={下载cl100k_base.tiktoken的所在的路径的绝对全路径}"
    ],
    ......
```


- **ceval**

| 参数           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| random\[可选\] | 随机抽样设置，由下面两个参数构成。<br>random_seed 随机种子\[必选\]<br>random_num 抽样数\[必选\]<br>如：--random_seed 0 --random_num 50<br>如果不设置此项，则为全量测试。 |
| test_jsonl     | /workspace/ModelZoo.LLM.Inference/dataset下提供了默认的ceval_val_cmcc.jsonl |
| timeout        | 超时设置，默认1200s |

```json
{
    ......
    "benchmark_cmds": {
        "ceval": {
            "type": "ceval",
            "command_base": "python3  run_ceval_client.py --model /models/DeepSeek-R1-0528-BF16-W8A8  --test_jsonl /workspace/ModelZoo.LLM.Inference/dataset/ceval_val_cmcc.jsonl --batch_size 64 --random_seed 0 --random_num 5"
        },
        "mmlu": {
            "type": "mmlu",
            "environment": ["mmlu_envs"],
            "command_base": "python3 bench_sglang.py --data_dir /models/acc/mmlu/data --nsub 60"
        }
    },
    "tasks": {
        "DeepSeek-R1-0528-BF16-W8A8" : {
            ......
            "benchmark": "ceval;mmlu"
            ......
        }
    }
    ......
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
　　│    └──  merge_perf_result.csv            # 本轮测试的性能测试结果汇总（若有）
　　│    └──  merge_acc_result.csv             # 本轮测试的精度测试结果汇总（若有）
　　│    └──  bench_record.log                 # 本轮测试的测试脚本执行记录和结果汇总日志
　　└── total_real_progress_file.json          # 全局进度控制文件
```

**层级3**: 每种任务各自目录，包含logs(实时日志)，result(任务结果相关)。result下目录结构根据任务类型不同有所差异，具体参考下面结构。

```plaintext
{模型名}/
　├── logs/
　│　 └── {任务名}_server{任务编号}_node{节点}_{ip}.log # 实时日志
　└── result/
　　　└── {任务名}-server{任务编号}/                   # 单任务结果
　　　 　　├── *.jsonl                                # 性能或精度测试结果
　　　 　　├── *.txt                                  # 性能或精度测试过程输出
　　　 　　└── *.csv                                  # 性能或精度结果指标提取
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
    "environments": {
        ......
  }
}
```
需要注意的是，所有任务执行时最大需使用几台机器，配置几台机器即可，否则，多余配置的机器也需要作为从节点启动。

## 4.3 配置任务信息

进入主节点容器内部，modelzoo目录就在 /workspace/ModelZoo.LLM.Inference

以DeepSeek-R1-W8A8为例，测试的任务是性能测试，那么修改 

 /workspace/ModelZoo.LLM.Inference/models/DeepSeek-R1-W8A8/config.json

```json
{
    "server_cmds": {
        "server_cmd": [
            ["python3 -m sglang.launch_server --trust-remote-code"],
            ["--model-path /models/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model"],
            ["--disable-radix-cache"],
            ["--attention-backend flashinfer"],
            ["--tp 16 --dp 8 --enable-dp-attention --enable-dp-lm-head"],
            [" --speculative-algorithm NEXTN --speculative-draft-model-path /models/DeepSeek-R1-NextN-Channel-INT8 --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --quantization w8a8_int8"]
        ]
    },
    "benchmark_cmds": {
        "random": {
            "command_base": "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --dataset-path /models/ShareGPT_V3_unfiltered_cleaned_split.json ",
            "input_output_len": ["3072/1024"],
            "max_concurrency": ["1", "16", "32", "64", "128"],
            "num_prompt_times": 5
        }
    },
    "tasks": {
        "test" : {
            "launch_server": "server_cmd",
            "benchmark": "random"
        }
    }
}
```

配置里面默认带的是本版本最佳性能参数，需要用户手动手改model_path，mtp model_path，ShareGPT_V3_unfiltered_cleaned_split.json（如有使用）的路径为镜像内部可以访问的路径，

如果想测试其他参数组合，可以在launch_server里面加，会自动组合生成测试结果，且json支持添加多个任务；需要注意的是，如果测试其他组合出现**out of memory**，需自行调小**mem-frac**的配置，查看当前使用mem-frac的大小方法：如果参数中有设置就是设置的值，如果参数中没设置，就在服务启动日志中搜索**mem_fraction_static**即可。

## 4.4 启动benchmark

```python
# (容器内)进入code目录
cd /workspace/ModelZoo.LLM.Inference/code
# 对mechines.json文件中除主节点以外的“所有”从节点执行（无论从节点在此任务中有没有使用到），以上面4.2中的配置信息为例，需要对ip为192.168.0.2的设备执行即可。 port 可自定义（保持主从一致），
python3 -m src.slave --local-ip 192.168.1.10 --port 20005 
# 必选参数：
--local-ip：当前节点ip，与mechines.json中配置的从节点ip保持一致
# 可选参数：
--port：从节点监听的端口后，必须和主节点一致，默认是20000

# 在主节点执行， port 可自定义（保持主从一致），可不配置，默认20000
python3 -m src.master --output-path ../outputs/ --tasks-config ../models/DeepSeek-R1-W8A8/config.json --machine-config ../models/mechines.json --port 20005


# 参数说明：
--machine-config：本次测试需要的机器信息
--output-path：结果输出的根目录，最好是外部挂载进容器的目录，防止容器删了结果丢失
--tasks-config：指定测试的配置文件或者目录，可以指定多个配置
--image-tag：镜像标签，用于结果区分，默认为空
--specify-test：从测试的配置文件中筛选出特定的任务类型执行，支持perf,mmlu,ceval，默认是全部执行，可选择多个
--port：socket的端口号，默认20000
--timeout：测试的超时时间，单位是秒，默认1200秒
--local-ip：当前节点ip，与mechines.json中配置的主节点ip保持一致

```

测试完成日志和结果都存放在output-path，结构说明见第3章 

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