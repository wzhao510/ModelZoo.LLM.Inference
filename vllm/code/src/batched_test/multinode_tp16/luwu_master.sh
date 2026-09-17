#!/bin/bash
# 陆吾 queue 模式 - master 容器启动脚本(主节点)
# 说明: queue 模式 start_script 前台串行执行, 跑完退出后平台自动释放机器, 不能 nohup 后台化
# 一次申请内顺序跑完 YAML 清单(configs/models_distributed_2nodes.yaml, 可用 MODEL_CONFIG 换)
# 里 default: true 的模型; 临时换其它模型: MODELS_SOURCE=env MODELS=<name>[,<name>...]
# 每个模型: 启动 rank0 -> 等健康 -> client 推理(含 long-text) -> vllm bench -> 停止
# 模型切换: 本机 + worker 都确认停止干净后才进入下一个(worker_ready 握手)
set -uo pipefail

# 脚本自身所在目录, /sw_home/lli/ModelZoo.LLM.Inference/...(同一份代码被放在
# 别的目录, 比如 ModelZoo.LLM.Inference.origin 副本时, 避免"跑 A 目录的脚本、cd 进
# B 目录用旧 config.sh"混用)
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$BASE/../requirements.txt"
JOB_TAG="${JOB_ID:-default}"

# 日志目录: 共享盘优先; 不可写(只读挂载/权限不足)时退到 /tmp, 保证启动日志一定存在
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

# 每次申请独立 run 目录(时间戳命名): 启动阶段和后续所有日志都写同一个 luwu_master.log
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BASE_LOG_DIR/$RUN_ID"
if ! { mkdir -p "$RUN_DIR" 2>/dev/null && [ -w "$RUN_DIR" ]; }; then
  RUN_DIR="/tmp/luwu_logs/$RUN_ID"
  mkdir -p "$RUN_DIR" 2>/dev/null || true
fi
LOG="$RUN_DIR/luwu_master.log"
FAILED_MARKER="$META_DIR/failed_${JOB_TAG}"
rm -f "$FAILED_MARKER" 2>/dev/null || true

# 输出同时落盘 + 发到平台日志; stdbuf 强制行缓冲, 否则平台日志会长时间只显示开头几行
TEE=(tee -a "$LOG")
command -v stdbuf >/dev/null 2>&1 && TEE=(stdbuf -oL -eL tee -a "$LOG")
exec > >("${TEE[@]}") 2>&1
say() { echo "[luwu-master] $(date '+%F %T') $*"; }

say "启动 host=$(hostname) ip=$(hostname -I 2>/dev/null | tr -s ' ' ',')"
say "JOB_ID=${JOB_ID:-<未注入>} cwd=$PWD MODEL_CONFIG=${MODEL_CONFIG:-<默认>} MODELS=${MODELS:-<默认>} LUWU_COMPILE=${LUWU_COMPILE:-0}"
say "日志文件: $LOG ${LOG_FALLBACK:+(共享盘 $LUWU_LOG_DIR 不可写, 已退到 /tmp)}"
say "环境: python3=$(command -v python3 2>/dev/null || echo 无) flock=$(command -v flock 2>/dev/null || echo 无)"

# 每一步都先打印再执行: 卡在哪一步, 平台日志最后一行就是哪一步
say "step1/5 进入代码目录 $BASE"
cd "$BASE" || { say "ERROR 进入代码目录失败(检查 /sw_home/lli 是否挂载): $BASE"; echo "cd-failed" > "$FAILED_MARKER" 2>/dev/null; exit 1; }

say "step2/5 加载模型清单 $BASE/config.sh"
if ! source ./config.sh; then
  say "ERROR 模型清单加载失败, 本次任务不会加载任何模型(原因见上面 [config] ERROR)"
  echo "config-load-failed: MODEL_CONFIG=${MODEL_CONFIG:-<默认>}" > "$FAILED_MARKER" 2>/dev/null
  exit 1
fi
if [ -z "${DIST_MODELS+x}" ]; then
  say "ERROR $BASE/config.sh 不是 YAML 版(没有 DIST_MODELS): 脚本和 config.sh 不属于同一份代码"
  say "      当前 BASE=$BASE, 请确认 start_script 里的目录与代码目录一致"
  echo "config-version-mismatch: $BASE/config.sh" > "$FAILED_MARKER" 2>/dev/null
  exit 1
fi
say "step2/5 OK: 清单 $(basename "$MODEL_CONFIG"), 共 ${#DIST_MODELS[@]} 个模型"

# 单实例锁(按 JOB_ID 区分, 保证平台重复拉起时锁稳定): 后启动的实例等待前一个完成
say "step3/5 取单实例锁 $META_DIR/lock_master_${JOB_TAG}"
LOCK="$META_DIR/lock_master_${JOB_TAG}"
exec 9>"$LOCK"
if ! flock -n 9; then
  say "检测到已有 master 实例在运行, 等待其完成..."
  flock 9
  say "前序实例已结束, 本实例直接退出"
  say "提示: 若两个节点都跑了 luwu_master.sh(同一个 JOB_ID), 只有先抢到锁的会真正跑, 另一个节点必须改跑 luwu_worker.sh"
  exit 0
fi

# run 目录已在启动时创建(日志从一开始就在里面), 这里只发布给 worker 保持一致
say "step4/5 发布 run 目录给 worker: $RUN_ID ($RUN_DIR)"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$META_DIR/current_run_${JOB_TAG}"

# 模型列表(顺序执行)默认由 YAML 清单决定: 取清单里 default: true 的模型。
# 需要临时用环境变量指定(逗号分隔)时显式设 MODELS_SOURCE=env。
MODELS_SOURCE="${MODELS_SOURCE:-yaml}"
MODELS_RAW="${MODELS:-}"
MODELS_FROM="YAML $MODEL_CONFIG 里 default: true 的模型"
if [ "$MODELS_SOURCE" = "env" ]; then
  # 兼容逗号分隔(陆吾 env_vars 注入)与空格分隔两种写法
  MODELS="${MODELS_RAW//,/ }"
  read -r -a MODELS <<<"$MODELS"
  MODELS_FROM="环境变量 MODELS=$MODELS_RAW"
else
  MODELS=("${DIST_DEFAULT_MODELS[@]}")
  if [ -n "$MODELS_RAW" ]; then
    say "提示: 已忽略环境变量 MODELS=$MODELS_RAW(模型列表走 YAML); 需要用它请加 MODELS_SOURCE=env"
  fi
fi
# 模型列表校验: 空列表 / 未知模型名都直接报错退出, 避免"没加载任何模型却显示执行完毕"
if [ "${#MODELS[@]}" -eq 0 ]; then
  say "ERROR 模型列表为空(来源: $MODELS_FROM); 清单 $MODEL_CONFIG 里 default: true 的模型: ${DIST_DEFAULT_MODELS[*]}"
  echo "empty-model-list: $MODELS_FROM" > "$FAILED_MARKER"
  exit 2
fi
UNKNOWN_MODELS=()
for _m in "${MODELS[@]}"; do
  [ -n "${MODEL_PATHS[$_m]:-}" ] || UNKNOWN_MODELS+=("$_m")
done
if [ "${#UNKNOWN_MODELS[@]}" -gt 0 ]; then
  say "ERROR 清单 $MODEL_CONFIG 里没有这些模型: ${UNKNOWN_MODELS[*]} (来源: $MODELS_FROM)"
  say "清单里的模型: ${DIST_MODELS[*]}"
  echo "unknown-models: ${UNKNOWN_MODELS[*]}" > "$FAILED_MARKER"
  exit 2
fi
rm -f "$FAILED_MARKER"
SHARED="$RUN_DIR/current_model"
WORKER_READY="$RUN_DIR/worker_ready"
rm -f "$SHARED" "$WORKER_READY"

say "step5/5 run=$RUN_ID 开始多模型顺序测试: ${MODELS[*]}(列表来源: $MODELS_FROM)"
say "平台注入环境变量:"
env | grep -E '^(MASTER_IP|WORKER_IP|VC_MASTER|VC_WORKER|SSH_PORT|NETWORK_CONFIG|GLOO|MCCL|JOB_ID)' || true

# master 地址: 优先取平台注入的 MASTER_IP, 否则 VC_MASTER_HOSTS
if [ -n "${MASTER_IP:-}" ]; then
  export MASTER_ADDR="$MASTER_IP"
elif [ -n "${VC_MASTER_HOSTS:-}" ]; then
  export MASTER_ADDR="$VC_MASTER_HOSTS"
fi
echo "[luwu-master] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT SERVE_PORT=$SERVE_PORT"

# 环境初始化(幂等)
bash /sw_home/lli/compile_env.sh >/dev/null 2>&1 || echo "[luwu-master] WARN compile_env.sh 失败, 继续"
pip install -r "$REQUIREMENTS" >/dev/null 2>&1 || true
pip install -q openai 2>/dev/null || true

# 可选: LUWU_COMPILE=1 时, 编译安装 luwu_apply 源码(mcoplib+vllm_metax)后再跑模型
if [ "${LUWU_COMPILE:-0}" = "1" ]; then
  echo "[luwu-master] LUWU_COMPILE=1, 执行源码编译安装..."
  if ! bash "$BASE/luwu_compile.sh"; then
    echo "[luwu-master] ERROR 编译安装失败, 终止"
    exit 1
  fi
fi

ALL_RC=0
for MODEL_NAME in "${MODELS[@]}"; do
  export MODEL_NAME
  export MODEL_RUN_DIR="$RUN_DIR/$MODEL_NAME"
  mkdir -p "$MODEL_RUN_DIR"
  echo ""
  echo "[luwu-master] $(date '+%F %T') ============ 开始模型: $MODEL_NAME ============"
  echo "$MODEL_NAME" > "$SHARED"

  # 启动 rank0(幂等: 若服务已在运行则跳过, 不因守卫报错而退出)
  RANK=0 MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$MODEL_RUN_DIR" bash start.sh \
    || echo "[luwu-master] WARN start.sh 返回 $?(可能服务已在运行), 继续等待健康"

  # 等待健康(每模型独立超时; Kimi bf16 加载约 25 分钟, 默认给 3600s)
  if bash check.sh "${CHECK_TIMEOUT:-3600}"; then
    # 推理客户端(类似单机 launch.py --infer, 含 long-text)
    MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$MODEL_RUN_DIR" bash client.sh
    CLIENT_RC=$?
    [ "$CLIENT_RC" -ne 0 ] && ALL_RC="$CLIENT_RC"
    echo "[luwu-master] $MODEL_NAME client 推理完成 exit=$CLIENT_RC"

    # 性能压测
    MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$MODEL_RUN_DIR" bash bench.sh
    RC=$?
    [ "$RC" -ne 0 ] && ALL_RC="$RC"
    echo "[luwu-master] $MODEL_NAME bench 完成 exit=$RC"
  else
    echo "[luwu-master] ERROR $MODEL_NAME 未在限时内就绪, 跳过"
    ALL_RC=1
  fi

  # 停止本机服务并确认清干净
  MODEL_NAME="$MODEL_NAME" bash stop.sh || true
  for i in {1..60}; do
    pgrep -f "VLLM::" >/dev/null 2>&1 || break
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 2
  done

  # 等待 worker 确认已停干净(worker_ready 握手), 两机都停掉再进下一个模型
  for i in {1..150}; do
    [ "$(cat "$WORKER_READY" 2>/dev/null)" = "$MODEL_NAME" ] && break
    sleep 2
  done
  if [ "$(cat "$WORKER_READY" 2>/dev/null)" = "$MODEL_NAME" ]; then
    echo "[luwu-master] $MODEL_NAME worker 已确认停止, 进入下一个模型"
  else
    echo "[luwu-master] WARN 等待 worker 确认超时, 仍继续下一个模型"
  fi
  rm -f "$WORKER_READY"
  sleep 10
done

echo "DONE_ALL" > "$SHARED"
echo "[luwu-master] $(date '+%F %T') 全部模型执行完毕, exit=$ALL_RC"
exit "$ALL_RC"
