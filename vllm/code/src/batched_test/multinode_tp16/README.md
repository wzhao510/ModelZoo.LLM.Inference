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
| `config.sh` | 通用配置：端口/TP/模型路径/环境变量 |
| `start.sh` | 启动本节点 vllm serve（`RANK=0`/`RANK=1`） |
| `check.sh` | 轮询 rank0 `/health` 直到就绪 |
| `bench.sh` | 对已启动的 rank0 服务跑 `vllm bench serve` 压测 |
| `status.sh` | 查看本节点 serve 进程与 rank0 健康状态 |
| `stop.sh` | 停止本节点 serve |
| `run_all.sh` | rank0 侧一键流程：等待就绪 -> 压测 -> 停止 |

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
export MODEL_NAME=DeepSeek-R1-0528-W8A8   # 可选: GLM-5.2-W8A8 / Kimi-K2.6-Int4

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

## 模型与路径

| 模型名 | 路径 | dtype |
| --- | --- | --- |
| DeepSeek-R1-0528-W8A8 | `/mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model/` | bfloat16 |
| GLM-5.2-W8A8 | `/mxstorage/pde_ai/models/llm/ChatGLM/GLM-5_2-W8A8/` | bfloat16 |
| Kimi-K2.6-Int4 | `/mxstorage/pde_ai/models/llm/Kimi/Kimi-K2.6-W8A8/` | float16 |

如需加载报错，可用 `DTYPE=bfloat16` 覆盖（Kimi 的 config.json 为 bfloat16）。

## 注意

- 脚本需在容器内执行（vllm 在容器内 `/opt/conda/bin/vllm`）。
- `--network=host` 容器共享宿主机网络，端口冲突时务必换端口。
- `stop.sh` 只影响本容器内的进程（PID namespace 隔离），不会误杀其他用户的容器。
- 若需 speculative（如 deepseek_mtp / mtp），在 `config.sh` 的 `MODEL_EXTRA_ARGS` 中追加
  `--speculative-config` 参数。
