#!/bin/bash
# 陆吾 queue 模式 - worker 容器启动脚本(从节点)
# 说明: rank1 headless 后台运行; 通过共享标记文件(current_model)跟随 master 逐个模型
# 关键: master 停止当前模型(TCPStore 8801 掉线)后, worker 主动杀掉 rank1 并清理,
#       写 worker_ready 通知 master 本机已停干净, 两机都停掉才进入下一个模型
set -uo pipefail

BASE="/sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16"
cd "$BASE" || exit 1
source ./config.sh || exit 1

# 日志目录: 按日期+时分秒命名(与 master 一致, 读 master 发布的名字)
export BASE_LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp16_luwu}"
mkdir -p "$BASE_LOG_DIR"
JOB_TAG="${JOB_ID:-default}"
META_DIR="$BASE_LOG_DIR/.luwu_meta"
mkdir -p "$META_DIR"

# 单实例锁(按 JOB_ID 区分): 平台重复拉起时, 后启动的 worker 等待前一个完成
LOCK="$META_DIR/lock_worker_${JOB_TAG}"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[luwu-worker] 检测到已有 worker 实例在运行, 等待其完成..."
  flock 9
  echo "[luwu-worker] 前序实例已结束, 本实例直接退出"
  exit 0
fi

# 等待 master 发布的日期目录名(worker 可能先于 master 启动, 最多等 120s)
RUN_ID=""
for i in {1..120}; do
  RUN_ID="$(cat "$META_DIR/current_run_${JOB_TAG}" 2>/dev/null || true)"
  [ -n "$RUN_ID" ] && break
  sleep 1
done
if [ -z "$RUN_ID" ]; then
  RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
  echo "[luwu-worker] WARN 未等到 master 发布的 RUN_ID, 用本地时间兜底: $RUN_ID"
fi
RUN_DIR="$BASE_LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/luwu_worker.log"
exec > >(tee -a "$LOG") 2>&1

# 模型列表(与 master 保持一致); 可通过环境变量 MODELS 覆盖, 默认同 master 的默认清单
if [ -n "${MODELS:-}" ]; then
  # 兼容逗号分隔(陆吾 env_vars 注入)与空格分隔两种写法
  MODELS="${MODELS//,/ }"
  read -r -a MODELS <<<"$MODELS"
else
  MODELS=("${DIST_DEFAULT_MODELS[@]}")
fi
SHARED="$RUN_DIR/current_model"
WORKER_READY="$RUN_DIR/worker_ready"

# rank1 存活判定: 优先用 start.sh 落盘的 pid 文件, 退化到 pgrep(匹配方式与 start.sh 一致)
rank1_alive() {
  if [ -n "${RANK1_PID:-}" ] && kill -0 "$RANK1_PID" 2>/dev/null \
     && grep -qa "vllm serve" "/proc/$RANK1_PID/cmdline" 2>/dev/null; then
    return 0
  fi
  if [ -n "${MODEL_PATH:-}" ] && pgrep -f "vllm serve ${MODEL_PATH}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if [ -n "${MASTER_IP:-}" ]; then
  export MASTER_ADDR="$MASTER_IP"
elif [ -n "${VC_MASTER_HOSTS:-}" ]; then
  export MASTER_ADDR="$VC_MASTER_HOSTS"
fi
echo "[luwu-worker] $(date '+%F %T') run=$RUN_ID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 环境初始化(幂等)
bash /sw_home/lli/compile_env.sh >/dev/null 2>&1 || echo "[luwu-worker] WARN compile_env.sh 失败, 继续"
pip install -r /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/requirements.txt >/dev/null 2>&1 || true

# 可选: LUWU_COMPILE=1 时, 安装 master 的编译产物(等共享 wheel); 等不到则本机编译
if [ "${LUWU_COMPILE:-0}" = "1" ]; then
  echo "[luwu-worker] LUWU_COMPILE=1, 等待/安装编译产物..."
  if ! bash "$BASE/luwu_compile.sh" --install; then
    echo "[luwu-worker] ERROR 编译产物安装失败, 终止"
    exit 1
  fi
fi

store_down=0
last=""
while true; do
  cur="$(cat "$SHARED" 2>/dev/null || true)"
  if [ "$cur" = "DONE_ALL" ]; then
    echo "[luwu-worker] $(date '+%F %T') master 已结束全部任务, worker 退出"
    bash stop.sh --all || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
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
    # 与 master 一致: 统一走 start.sh(RANK=1 headless), 保证两机的 serve 参数完全一致
    # (TP/DP/PP/NNODES/max-model-len/gpu-mem/max-num-seqs/dtype/extra_args 都按模型清单
    #  configs/models_distributed_tp16.yaml 里该模型的 serve_config 取值)
    RANK=1 MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$MODEL_RUN_DIR" bash start.sh \
      || echo "[luwu-worker] WARN start.sh 返回 $?(可能已有 rank1 在运行), 继续等待/监控"
    RANK1_PID="$(cat "$MODEL_RUN_DIR/rank1.pid" 2>/dev/null || true)"
    if [ -z "$RANK1_PID" ]; then
      RANK1_PID="$(pgrep -f "vllm serve ${MODEL_PATH}" 2>/dev/null | head -n1 || true)"
    fi
    echo "[luwu-worker] rank1 pid=${RANK1_PID:-unknown} log=$MODEL_RUN_DIR/rank1_serve.log"

    # 监控: rank1 退出 / master 切换标记 / master store 掉线(视为 master 已停)时停止当前 rank1
    store_down=0
    while rank1_alive; do
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

    # 停止 rank1 并清理残留(与 master 一致走 stop.sh), 等待 GPU 内存释放
    MODEL_NAME="$MODEL_NAME" bash stop.sh || true
    if [ -n "${RANK1_PID:-}" ]; then
      kill -9 "$RANK1_PID" 2>/dev/null || true
    fi
    for i in {1..30}; do
      pgrep -f "VLLM::" >/dev/null 2>&1 || break
      pkill -9 -f "VLLM::" 2>/dev/null || true
      sleep 2
    done
    sleep 10
    # 通知 master: 本机已停干净
    echo "$MODEL_NAME" > "$WORKER_READY"
    echo "[luwu-worker] $MODEL_NAME 已清理, 通知 master (worker_ready)"
  fi
  sleep 3
done
