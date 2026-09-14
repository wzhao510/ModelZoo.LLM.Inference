#!/bin/bash
# 启动本节点 vllm serve (TP16 多机的一个 rank)
# 模型清单来自 configs/models_distributed_tp16.yaml (见 config.sh)
# 用法: RANK=0 bash start.sh   # rank0 (服务节点 10.13.81.57)
#       RANK=1 bash start.sh   # rank1 (headless 节点 10.13.81.58)
# 可选: MODEL_NAME=GLM-5.2-W8A8 SERVE_PORT=8010 MASTER_PORT=8802 bash start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

RANK="${RANK:?请设置 RANK=0 或 RANK=1}"
MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]:-}"
if [ -z "$MODEL_PATH" ]; then
  echo "[start] 未知模型 '$MODEL_NAME', 清单: $MODEL_CONFIG" >&2
  echo "[start] 可选模型: ${DIST_MODELS[*]}" >&2
  exit 2
fi

# 每模型参数优先, 未配置时回落到 config.sh 的全局默认值
DTYPE="${DTYPE:-${MODEL_DTYPES[$MODEL_NAME]:-bfloat16}}"
EXTRA="${MODEL_EXTRA_ARGS[$MODEL_NAME]:-}"
MODEL_ENV="${MODEL_ENVS[$MODEL_NAME]:-}"
TP_M="${MODEL_TP[$MODEL_NAME]:-${TP:-1}}"
DP_M="${MODEL_DP[$MODEL_NAME]:-${DP:-1}}"
PP_M="${MODEL_PP[$MODEL_NAME]:-${PP:-1}}"
GPU_MEM_M="${MODEL_GPU_MEM[$MODEL_NAME]:-${GPU_MEM_UTIL:-0.9}}"
MAX_LEN_M="${MODEL_MAX_MODEL_LEN[$MODEL_NAME]:-${MAX_MODEL_LEN:-4096}}"
MAX_SEQS_M="${MODEL_MAX_NUM_SEQS[$MODEL_NAME]:-${MAX_NUM_SEQS:-64}}"

# 节点数: 由该模型卡数(tp*dp*pp)与本机卡数推导, 也可用 NNODES 覆盖
GPU_COUNT=$((TP_M * DP_M * PP_M))
AUTO_NNODES=$(( (GPU_COUNT + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
if (( AUTO_NNODES < 1 )); then
  AUTO_NNODES=1
fi
NNODES_M="${NNODES:-$AUTO_NNODES}"

# 模型专属环境变量(清单里的 extra_env)
if [ -n "$MODEL_ENV" ]; then
  eval "export $MODEL_ENV"
fi

RUN_DIR="${MODEL_RUN_DIR:-$BASE_LOG_DIR/$MODEL_NAME}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/rank${RANK}_serve.log"

case "$RANK" in
  0)
    RANK_ARGS=(--node-rank 0 --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
               --host "$SERVE_HOST" --port "$SERVE_PORT")
    ;;
  1)
    RANK_ARGS=(--node-rank 1 --master-addr "$MASTER_ADDR" --headless --master-port "$MASTER_PORT")
    ;;
  *)
    echo "[start] RANK 必须为 0 或 1" >&2
    exit 2
    ;;
esac

if pgrep -f "vllm serve ${MODEL_PATH}" >/dev/null 2>&1; then
  echo "[start] 已检测到 ${MODEL_NAME} 的 vllm serve 在运行, 请先执行 stop.sh" >&2
  pgrep -f "vllm serve ${MODEL_PATH}" | head -n1 > "$RUN_DIR/rank${RANK}.pid" 2>/dev/null || true
  exit 3
fi

CMD=(vllm serve "$MODEL_PATH" --trust-remote-code --distributed-executor-backend mp
     --max-model-len "$MAX_LEN_M" --gpu-memory-utilization "$GPU_MEM_M"
     -tp "$TP_M" -dp "$DP_M" -pp "$PP_M" --no-enable-prefix-caching
     --max-num-seqs "$MAX_SEQS_M"
     --nnodes "$NNODES_M" --dtype "$DTYPE" "${RANK_ARGS[@]}")
# EXTRA 已在 load_models.py 里按 shell 规则转义(含引号的 --speculative-config 等)
if [ -n "$EXTRA" ]; then
  eval "CMD+=( $EXTRA )"
fi

echo "[start] $(date '+%F %T') rank=$RANK model=$MODEL_NAME tp=$TP_M dp=$DP_M pp=$PP_M nnodes=$NNODES_M"
echo "[start] cmd: ${CMD[*]}"
nohup "${CMD[@]}" >"$LOG" 2>&1 &
SERVE_PID=$!
# 记录 pid 供调用方(如 luwu_worker.sh)监控/清理, 与 master 走同一套启动逻辑
echo "$SERVE_PID" > "$RUN_DIR/rank${RANK}.pid"
echo "[start] pid=$SERVE_PID log=$LOG"
