#!/bin/bash
# 陆吾 queue 模式 - worker 容器启动脚本(从节点)
# 说明: rank1 headless 后台运行; 通过共享标记文件(current_model)跟随 master 逐个模型
# 关键: master 停止当前模型(TCPStore 8801 掉线)后, worker 主动杀掉 rank1 并清理,
#       写 worker_ready 通知 master 本机已停干净, 两机都停掉才进入下一个模型
set -uo pipefail

# 脚本自身所在目录(理由同 luwu_master.sh: 不写死绝对路径, 避免新旧代码目录混用)
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$BASE/../requirements.txt"
JOB_TAG="${JOB_ID:-default}"

# 日志目录: 共享盘优先; 不可写时退到 /tmp(与 master 同规则)
BASE_LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp16_luwu}"
LOG_FALLBACK=""
if ! { mkdir -p "$BASE_LOG_DIR" 2>/dev/null && [ -w "$BASE_LOG_DIR" ]; }; then
  LOG_FALLBACK="yes"
  BASE_LOG_DIR="/tmp/luwu_logs"
  mkdir -p "$BASE_LOG_DIR" 2>/dev/null || true
fi
export BASE_LOG_DIR
META_DIR="$BASE_LOG_DIR/.luwu_meta"
mkdir -p "$META_DIR" 2>/dev/null || true
# master 还没发布 run 目录时, 日志先落在 meta 目录; 拿到 RUN_ID 后挪进 run 目录并改名
# luwu_worker.log(tee 持有该 inode, 挪动后后续输出继续写到新路径, 全程一个文件)
LOG="$META_DIR/boot_worker_${JOB_TAG}.log"
MASTER_LOG_HINT="tail -n50 \$(ls -t $BASE_LOG_DIR/run_*/luwu_master.log 2>/dev/null | head -1)"
FAILED_MARKER="$META_DIR/failed_${JOB_TAG}"

# 输出同时落盘 + 发到平台日志; stdbuf 强制行缓冲, 否则平台日志会长时间只显示开头几行
TEE=(tee -a "$LOG")
command -v stdbuf >/dev/null 2>&1 && TEE=(stdbuf -oL -eL tee -a "$LOG")
exec > >("${TEE[@]}") 2>&1
say() { echo "[luwu-worker] $(date '+%F %T') $*"; }

say "启动 host=$(hostname) ip=$(hostname -I 2>/dev/null | tr -s ' ' ',')"
say "JOB_ID=${JOB_ID:-<未注入>} cwd=$PWD MODEL_CONFIG=${MODEL_CONFIG:-<默认>} MODELS=${MODELS:-<默认>} LUWU_COMPILE=${LUWU_COMPILE:-0}"
say "日志文件: $LOG ${LOG_FALLBACK:+(共享盘 $LUWU_LOG_DIR 不可写, 已退到 /tmp)}(拿到 master 的 run 目录后变成 run_*/luwu_worker.log)"

say "step1/6 进入代码目录 $BASE"
cd "$BASE" || { say "ERROR 进入代码目录失败(检查 /sw_home/lli 是否挂载): $BASE"; exit 1; }

say "step2/6 加载模型清单 $BASE/config.sh"
if ! source ./config.sh; then
  say "ERROR 模型清单加载失败, 本节点不会起 rank1(原因见上面 [config] ERROR)"
  exit 1
fi
if [ -z "${DIST_MODELS+x}" ]; then
  say "ERROR $BASE/config.sh 不是 YAML 版(没有 DIST_MODELS): 脚本和 config.sh 不属于同一份代码"
  say "      当前 BASE=$BASE, 请确认 start_script/start_script_worker 里的目录与代码目录一致"
  exit 1
fi
say "step2/6 OK: 清单 $(basename "$MODEL_CONFIG"), 共 ${#DIST_MODELS[@]} 个模型, 本次执行 ${#DIST_RUN_MODELS[@]} 个"

# 单实例锁(按 JOB_ID 区分): 平台重复拉起时, 后启动的 worker 等待前一个完成
say "step3/6 取单实例锁 $META_DIR/lock_worker_${JOB_TAG}"
LOCK="$META_DIR/lock_worker_${JOB_TAG}"
exec 9>"$LOCK"
if ! flock -n 9; then
  say "检测到已有 worker 实例在运行, 等待其完成..."
  flock 9
  say "前序实例已结束, 本实例直接退出"
  exit 0
fi

# 等待 master 发布的日期目录名(worker 可能先于 master 启动);
# 同时监控 master 的失败标记, 避免 master 已经失败而 worker 一直挂着
say "step4/6 等 master 发布 RUN_ID(同时监控 master 失败标记)"
RUN_ID_WAIT="${RUN_ID_WAIT:-300}"
RUN_ID=""
for ((i = 1; i <= RUN_ID_WAIT; i++)); do
  RUN_ID="$(cat "$META_DIR/current_run_${JOB_TAG}" 2>/dev/null || true)"
  [ -n "$RUN_ID" ] && break
  if [ -f "$FAILED_MARKER" ]; then
    say "ERROR master 启动失败($(cat "$FAILED_MARKER" 2>/dev/null)): 本次任务不会加载模型"
    say "排查: $MASTER_LOG_HINT"
    exit 3
  fi
  if (( i % 30 == 0 )); then
    say "... 已等待 master 发布 RUN_ID ${i}s"
  fi
  sleep 1
done
if [ -z "$RUN_ID" ]; then
  say "ERROR 等待 master 发布 RUN_ID 超时(${RUN_ID_WAIT}s), 本节点退出"
  say "排查: 确认另一个节点跑的是 luwu_master.sh; master 日志: $MASTER_LOG_HINT"
  say "      本节点日志: $LOG"
  exit 3
fi
RUN_DIR="$BASE_LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR" 2>/dev/null || true
# 启动阶段(含 step 打点)的日志并入 master 的 run 目录, 与后续输出合成一个 luwu_worker.log
NEW_LOG="$RUN_DIR/luwu_worker.log"
if mv -f "$LOG" "$NEW_LOG" 2>/dev/null; then
  LOG="$NEW_LOG"
fi
say "step4/6 OK: run=$RUN_ID 日志=$LOG"
say "step5/6 跟随 master 的模型标记, 需要时启动 rank1"

# 模型列表(与 master 保持一致)默认由 YAML 清单决定: 取清单里 default: true 的模型。
# 需要临时用环境变量指定(逗号分隔)时显式设 MODELS_SOURCE=env。
MODELS_SOURCE="${MODELS_SOURCE:-yaml}"
MODELS_RAW="${MODELS:-}"
MODELS_FROM="YAML $MODEL_CONFIG 里列出的模型"
if [ "$MODELS_SOURCE" = "env" ]; then
  # 兼容逗号分隔(陆吾 env_vars 注入)与空格分隔两种写法
  MODELS="${MODELS_RAW//,/ }"
  read -r -a MODELS <<<"$MODELS"
  MODELS_FROM="环境变量 MODELS=$MODELS_RAW"
else
  MODELS=("${DIST_RUN_MODELS[@]}")
  if [ -n "$MODELS_RAW" ]; then
    say "提示: 已忽略环境变量 MODELS=$MODELS_RAW(模型列表走 YAML); 需要用它请加 MODELS_SOURCE=env"
  fi
fi
say "本次跟随的模型($MODELS_FROM): ${MODELS[*]}"
# 与 master 同样的校验: 空列表/未知模型名直接报错退出
if [ "${#MODELS[@]}" -eq 0 ]; then
  say "ERROR 模型列表为空(来源: $MODELS_FROM); 清单 $MODEL_CONFIG 里列出的模型: ${DIST_RUN_MODELS[*]}"
  exit 2
fi
UNKNOWN_MODELS=()
for _m in "${MODELS[@]}"; do
  [ -n "${MODEL_PATHS[$_m]:-}" ] || UNKNOWN_MODELS+=("$_m")
done
if [ "${#UNKNOWN_MODELS[@]}" -gt 0 ]; then
  say "ERROR 清单 $MODEL_CONFIG 里没有这些模型: ${UNKNOWN_MODELS[*]} (来源: $MODELS_FROM)"
  say "清单里的模型: ${DIST_MODELS[*]}"
  exit 2
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

# 环境初始化(幂等): 输出落盘, 成功/失败都明确打点, 避免"看不清有没有执行"
say "step6/6 环境初始化: compile_env.sh + pip 依赖"
if bash /sw_home/lli/compile_env.sh > "$RUN_DIR/compile_env.log" 2>&1; then
  say "compile_env.sh OK (输出: $RUN_DIR/compile_env.log)"
else
  RC_ENV=$?
  say "WARN compile_env.sh 失败 exit=$RC_ENV (输出: $RUN_DIR/compile_env.log), 继续"
fi
if pip install -r "$REQUIREMENTS" > "$RUN_DIR/pip_requirements.log" 2>&1; then
  say "requirements 安装 OK"
else
  say "WARN requirements 安装失败 (输出: $RUN_DIR/pip_requirements.log), 继续"
fi

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
  # master 早期失败(清单加载/模型名校验)时不干等: 直接停干净退出
  if [ -f "$FAILED_MARKER" ]; then
    echo "[luwu-worker] ERROR master 失败($(cat "$FAILED_MARKER" 2>/dev/null)), 停止本节点"
    echo "[luwu-worker] 排查: $MASTER_LOG_HINT"
    bash stop.sh --all || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    exit 3
  fi
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
    #  configs/models_distributed_2nodes.yaml 里该模型的 serve_config 取值)
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
