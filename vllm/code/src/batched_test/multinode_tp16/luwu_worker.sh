#!/bin/bash
# 陆吾 queue 模式 - worker 容器启动脚本(从节点)
# 说明: rank1 headless 后台运行; 通过共享标记文件(current_model)跟随 master 逐个模型
# 关键: master 停止当前模型(TCPStore 8801 掉线)后, worker 主动杀掉 rank1 并清理,
#       写 worker_ready 通知 master 本机已停干净, 两机都停掉才进入下一个模型
set -uo pipefail

BASE="/sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16"
cd "$BASE" || exit 1
source ./config.sh

# 与 master 一致的 run 目录(用平台注入的 JOB_ID)
RUN_ID="${JOB_ID:-run_$(date +%Y%m%d_%H%M%S)}"
export BASE_LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp16_luwu}"
RUN_DIR="$BASE_LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/luwu_worker.log"
exec > >(tee -a "$LOG") 2>&1

# 单实例锁: 平台重复拉起时, 后启动的 worker 等待前一个完成
LOCK="$RUN_DIR/luwu_worker.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[luwu-worker] 检测到已有 worker 实例在运行, 等待其完成..."
  flock 9
  echo "[luwu-worker] 前序实例已结束, 本实例直接退出"
  exit 0
fi

# 模型列表(与 master 保持一致); 可通过环境变量覆盖
if [ -n "${MODELS:-}" ]; then
  read -r -a MODELS <<<"$MODELS"
else
  MODELS=(DeepSeek-R1-0528-W8A8 GLM-5.2-W8A8 Kimi-K2.6-Int4)
fi
SHARED="$RUN_DIR/current_model"
WORKER_READY="$RUN_DIR/worker_ready"

if [ -n "${MASTER_IP:-}" ]; then
  export MASTER_ADDR="$MASTER_IP"
elif [ -n "${VC_MASTER_HOSTS:-}" ]; then
  export MASTER_ADDR="$VC_MASTER_HOSTS"
fi
echo "[luwu-worker] $(date '+%F %T') run=$RUN_ID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 环境初始化(幂等)
bash /sw_home/lli/compile_env.sh >/dev/null 2>&1 || echo "[luwu-worker] WARN compile_env.sh 失败, 继续"
pip install -r /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/requirements.txt >/dev/null 2>&1 || true

store_down=0
last=""
while true; do
  cur="$(cat "$SHARED" 2>/dev/null || true)"
  if [ "$cur" = "DONE_ALL" ]; then
    echo "[luwu-worker] $(date '+%F %T') master 已结束全部任务, worker 退出"
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    exit 0
  fi
  if [ -z "$cur" ] || [ "$cur" = "$last" ]; then
    store_down=0
    sleep 5
    continue
  fi
  # 校验是否在模型列表内(过滤其他杂信号)
  in_list=0
  for m in "${MODELS[@]}"; do
    [ "$m" = "$cur" ] && in_list=1
  done
  if [ "$in_list" = "1" ]; then
    last="$cur"
    MODEL_NAME="$cur"
    export MODEL_NAME
    MODEL_RUN_DIR="$RUN_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_RUN_DIR"
    echo "[luwu-worker] $(date '+%F %T') ============ 启动 rank1: $MODEL_NAME ============"
    # 等 master 的 TCPStore(8801)就绪后再启动, 避免连接超时
    for i in {1..300}; do
      if (echo > /dev/tcp/"$MASTER_ADDR"/"$MASTER_PORT") 2>/dev/null; then
        break
      fi
      sleep 1
    done
    MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
    DTYPE="${DTYPE:-${MODEL_DTYPES[$MODEL_NAME]:-bfloat16}}"
    nohup vllm serve "$MODEL_PATH" --trust-remote-code --distributed-executor-backend mp \
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      -tp "$TP" -dp "$DP" -pp "$PP" --no-enable-prefix-caching --max-num-seqs "$MAX_NUM_SEQS" \
      --nnodes "$NNODES" --dtype "$DTYPE" \
      --node-rank 1 --master-addr "$MASTER_ADDR" --headless --master-port "$MASTER_PORT" \
      > "$MODEL_RUN_DIR/rank1_serve.log" 2>&1 &
    RANK1_PID=$!
    echo "[luwu-worker] rank1 pid=$RANK1_PID log=$MODEL_RUN_DIR/rank1_serve.log"

    # 监控: rank1 退出 / master 切换标记 / master store 掉线(视为 master 已停)时停止当前 rank1
    store_down=0
    while kill -0 "$RANK1_PID" 2>/dev/null; do
      cur2="$(cat "$SHARED" 2>/dev/null || true)"
      if [ "$cur2" = "DONE_ALL" ]; then
        echo "[luwu-worker] master 已结束全部任务"
        break
      fi
      if [ -n "$cur2" ] && [ "$cur2" != "$last" ]; then
        echo "[luwu-worker] master 标记已切换($cur2), 停止当前 rank1"
        break
      fi
      if (echo > /dev/tcp/"$MASTER_ADDR"/"$MASTER_PORT") 2>/dev/null; then
        store_down=0
      else
        store_down=$((store_down + 1))
        # 约 60s 不可达, 判定 master 已停止当前模型
        if [ "$store_down" -ge 6 ]; then
          echo "[luwu-worker] master TCPStore 掉线, 判定当前模型已结束"
          break
        fi
      fi
      sleep 10
    done

    # 停止 rank1 并清理残留, 等待 GPU 内存释放
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    kill -9 "$RANK1_PID" 2>/dev/null || true
    sleep 10
    # 通知 master: 本机已停干净
    echo "$MODEL_NAME" > "$WORKER_READY"
    echo "[luwu-worker] $MODEL_NAME 已清理, 通知 master (worker_ready)"
  fi
  sleep 3
done
