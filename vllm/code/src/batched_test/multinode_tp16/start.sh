#!/bin/bash
# 启动本节点 vllm serve (TP16 多机的一个 rank)
# 用法: RANK=0 bash start.sh   # rank0 (服务节点 10.13.81.57)
#       RANK=1 bash start.sh   # rank1 (headless 节点 10.13.81.58)
# 可选: MODEL_NAME=GLM-5.2-W8A8 SERVE_PORT=8010 MASTER_PORT=8802 bash start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

RANK="${RANK:?请设置 RANK=0 或 RANK=1}"
MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
DTYPE="${DTYPE:-${MODEL_DTYPES[$MODEL_NAME]:-bfloat16}}"
EXTRA="${MODEL_EXTRA_ARGS[$MODEL_NAME]:-}"

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
     --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
     -tp "$TP" -dp "$DP" -pp "$PP" --no-enable-prefix-caching --max-num-seqs "$MAX_NUM_SEQS"
     --nnodes "$NNODES" --dtype "$DTYPE" "${RANK_ARGS[@]}")
# shellcheck disable=SC2206
CMD=("${CMD[@]}" $EXTRA)

echo "[start] $(date '+%F %T') rank=$RANK model=$MODEL_NAME"
echo "[start] cmd: ${CMD[*]}"
nohup "${CMD[@]}" >"$LOG" 2>&1 &
SERVE_PID=$!
# 记录 pid 供调用方(如 luwu_worker.sh)监控/清理, 与 master 走同一套启动逻辑
echo "$SERVE_PID" > "$RUN_DIR/rank${RANK}.pid"
echo "[start] pid=$SERVE_PID log=$LOG"
