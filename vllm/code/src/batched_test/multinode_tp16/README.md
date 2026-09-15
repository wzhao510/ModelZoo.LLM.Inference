# 多机（双机）TP16 测试脚本

用于在 2 台 MXC500X 节点（各 8 卡）上以 `tp=16` 跑多机 vLLM 推理/压测。

目标节点：
- rank0（服务节点）：`10.13.81.57`
- rank1（headless 节点）：`10.13.81.58`

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `docker_run.sh` | 创建容器（vllm 0.25.0-maca .103 镜像） |
| `setup.sh` | 容器内初始化：compile_env + batched_test 依赖 |
| `config.sh` | 通用配置：端口/全局默认参数/环境变量；模型清单从 YAML 加载 |
| `load_models.py` | 解析模型清单 YAML，生成 bash 数组供 `config.sh` 加载 |
| `start.sh` | 启动本节点 vllm serve（`RANK=0`/`RANK=1`） |
| `check.sh` | 轮询 rank0 `/health` 直到就绪 |
| `bench.sh` | 对已启动的 rank0 服务跑 `vllm bench serve` 压测 |
| `status.sh` | 查看本节点 serve 进程与 rank0 健康状态 |
| `stop.sh` | 停止本节点 serve |
| `run_all.sh` | rank0 侧一键流程：等待就绪 -> 压测 -> 停止 |
| `pyspy_watch.sh` | 启动期抓栈看门狗（py-spy dump + /proc 快照），由 `start.sh` 后台自动拉起 |

## 启动慢 / 卡住时怎么定位（pyspy_watch.sh）

`start.sh` 启动 serve 后会默认后台拉起 `pyspy_watch.sh`（`PYSPY_DUMP=0` 可关闭），
在启动到就绪期间每 30s 抓一轮，输出落在本模型的 run 目录下（双机各自一份）：

```
$MODEL_RUN_DIR/pyspy_rank0/          # rank1 同名 pyspy_rank1/
├── index.txt                        # 每轮: 耗时/进程数/py-spy 状态/日志末尾一行
├── watch.log                        # 看门狗自身日志(是否装上 py-spy 等)
├── round_01_t000010s/
│   ├── ps.txt                       # 各进程 CPU/RSS/状态/父子关系
│   ├── wchan_<pid>.txt              # 在算还是在等锁/IO
│   └── pyspy_<pid>_VLLM::Worker_TP0.txt   # Python 栈(卡在哪一行)
└── round_02_t000040s/ ...
```

排查方式：在 `index.txt` 里找到"日志末尾一行长时间不变"的那几轮，看对应轮次目录里的
`pyspy_*.txt`（Python 栈）和 `wchan_*.txt`（是否在 futex/socket 上等），就能定位卡点。

其他用法：

```bash
bash pyspy_watch.sh --once --out /tmp/dump          # 服务已经在跑, 立刻抓一次
PYSPY_INTERVAL=10 PYSPY_MAX_MIN=40 RANK=0 bash start.sh   # 加密采样
PYSPY_NATIVE=1 RANK=0 bash start.sh                 # 额外抓 native(C)栈, 更慢
```

注意：容器里 attach 需要 `--privileged` 或 `--cap-add=SYS_PTRACE`（本目录
`docker_run.sh` 用的是 `--privileged=true`，没问题）。没装 py-spy 时看门狗会尝试
`pip install py-spy`（离线环境设 `PYSPY_AUTO_INSTALL=0`），装不上则退化为只采
`ps`/`wchan` 快照。

## 快速开始（新容器）

```bash
# 1. 两台机器分别创建容器（10.13.81.57 与 10.13.81.58）
bash /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16/docker_run.sh

# 2. 两台机器容器内分别初始化环境
docker exec -it vllm_025_lli_0820 bash -c 'bash /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16/setup.sh'
```

## 跑双机 TP16

```bash
cd /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16
export MODEL_NAME=DeepSeek-R1-0528-W8A8   # 取模型清单里的 name, 见下方“模型清单”

# 10.13.81.58 容器内: 启动 rank1 (headless)
RANK=1 bash start.sh

# 10.13.81.57 容器内: 启动 rank0 (服务端)
RANK=0 bash start.sh

# 10.13.81.57 容器内: 等待就绪并压测
bash check.sh 3600
bash bench.sh

# 结束: 两台机器容器内分别停止
bash stop.sh
```

或者 rank0 侧一键执行（前提是 rank1 已先启动）：

```bash
MODEL_NAME=GLM-5.2-W8A8 bash run_all.sh          # 压测完自动停止
MODEL_NAME=Kimi-K2.6-Int4 KEEP_SERVER=1 bash run_all.sh   # 保留服务
```

## 常用覆盖项

端口被占用时通过环境变量覆盖（其他用户可能占用 8000/8800）：

```bash
export SERVE_PORT=8010 MASTER_PORT=8802
RANK=0 bash start.sh
```

其他可覆盖项：`TP/DP/PP/NNODES/MAX_MODEL_LEN/GPU_MEM_UTIL/MAX_NUM_SEQS/DTYPE`。

压测参数：`NUM_PROMPTS/MAX_CONCURRENCY/INPUT_LEN/OUTPUT_LEN/RESULT_DIR`。

## 模型清单（YAML 驱动）

模型列表放在 `../configs/models_distributed_tp16.yaml`，格式与
`../configs/models_single_QA_required.yaml` 一致，每个模型的 `serve_config`
自带 `tp/dp/pp/dtype/extra_args`：

```yaml
- name: GLM-4.5
  model_path: /mxstorage/pde_ai/models/llm/ChatGLM/GLM-4.5
  serve_config:
    tp: 8
    dp: 1
    pp: 2
    extra_args:
      --dtype: bfloat16
  infer_type:
  - text-only
  benchmark:
    bench_param: configs/bench_params/bench_high.json
    sweep_num_runs: 1
  extra_env:
```

要点：

- 卡数 = `tp*dp*pp`，节点数 `NNODES = 卡数 / GPUS_PER_NODE`（默认 8，可覆盖），
  所以新增模型只改 YAML 即可，不用动脚本。
- 现在清单里有 16 个模型：原 `config.sh` 的 3 个（`default: true`，多机 queue 任务默认跑）
  + `Kimi-K2.5-FP8`（C600U）+ 由 `configs/model.yaml` 迁入的 12 个大模型
  （`tp*dp*pp > 8`，单机 8 卡跑不动）。
- 选其它模型：`MODEL_NAME=GLM-4.5 bash start.sh`；queue 任务用
  `MODELS=GLM-5-W8A8,DeepSeek-V3.2-Exp` 覆盖默认列表。
- 换其它清单：`MODEL_CONFIG=/path/to/other.yaml bash start.sh`；相对路径按
  `batched_test/` 解析（`MODEL_CONFIG=configs/model.yaml` 与绝对路径等价）。
  路径不存在时脚本会打印尝试过的路径和 `configs/` 下可用清单后退出。
- 解析清单需要能 `import yaml` 的 python：脚本依次尝试 `MODELS_PYTHON` →
  `PATH` 里的 `python3` → `$CONDA_PREFIX/bin/python3` → `/opt/conda/bin/python3`。
  容器里 `/usr/bin/python3` 没有 PyYAML，`PATH` 缺 conda 时用
  `MODELS_PYTHON=/opt/conda/bin/python3` 指定即可。
- `MODELS` 里的模型名必须在清单里，且不能解析成空列表（拼错名字/`MODELS=,`/
  `MODELS` 为空串会直接报错退出并打印清单里的可用模型名），
  避免"一个模型都没加载就显示执行完毕"。
- 想看清单里全部模型名：
  `python3 load_models.py ../configs/models_distributed_tp16.yaml | head -1`。

## 排障（任务启动即结束）

多机 queue 任务"没加载模型就直接结束"时，按下面顺序看共享盘上的日志：

| 现象 | 日志/位置 | 原因 |
| --- | --- | --- |
| master/worker 秒退，容器 stdout 只有一行 `[config] ...` | `$BASE_LOG_DIR/run_<时间戳>/luwu_boot_master_<JOB_ID>.log`（worker 同名 boot log 在 master 的同一个 run 目录里） | 模型清单加载失败：`MODEL_CONFIG` 路径不对 / python 缺 PyYAML / YAML 语法错 |
| 日志只有 `开始多模型顺序测试:`（列表为空）后立刻 `全部模型执行完毕` | master 日志 | `MODELS` 解析成空列表 |
| 日志报 `清单里没有这些模型` | master/worker 日志 | `MODELS` 里写了清单中不存在的模型名（改名/迁到别的清单了） |
| worker 报 `master 启动失败` / `等待 master 发布 RUN_ID 超时` | worker 日志，以及 `$BASE_LOG_DIR/.luwu_meta/boot_worker_<JOB_ID>.log` | 另一节点没跑 `luwu_master.sh`，或 master 早期失败（看 `FAILED_MARKER`） |
| 一个节点打印 `前序实例已结束, 本实例直接退出` | 该节点日志 | 两个节点跑了同一份 `luwu_master.sh`（同 JOB_ID 抢锁），从节点应改跑 `luwu_worker.sh` |

失败标记写在 `$BASE_LOG_DIR/.luwu_meta/failed_<JOB_ID>`（内容为失败原因），
`$BASE_LOG_DIR` 默认 `/sw_home/lli/model_test/tp16_luwu`。

日志目录约定：

| 任务 | 日志目录 | 说明 |
| --- | --- | --- |
| 多机（tp16/tp32）| `/sw_home/lli/model_test/tp16_luwu/run_<时间戳>/` | 每次启动一个 run 目录：`luwu_boot_master_<JOB_ID>.log`、`luwu_master.log`、`luwu_boot_worker_<JOB_ID>.log`、`luwu_worker.log`、每个模型的 `rank*_serve.log` 都在里面 |
| 单机 | `/sw_home/lli/model_test/tp8_luwu/run_<时间戳>/` | `luwu_single.log`；`launch.py --infer` 自己的产物仍在 `/sw_home/lli/model_test/<时间戳>/` |

（可用 `LUWU_LOG_DIR` 覆盖上面两个默认根目录。）

## 注意

- 脚本需在容器内执行（vllm 在容器内 `/opt/conda/bin/vllm`）。
- `--network=host` 容器共享宿主机网络，端口冲突时务必换端口。
- `stop.sh` 只影响本容器内的进程（PID namespace 隔离），不会误杀其他用户的容器。
- 若需 speculative（如 deepseek_mtp / mtp），在该模型 YAML 的 `extra_args` 里追加
  `--speculative-config` 参数。
