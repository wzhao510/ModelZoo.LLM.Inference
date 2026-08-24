#!/bin/bash
# 陆吾 queue 模式 - master 容器启动脚本(主节点)
# 说明: queue 模式 start_script 前台串行执行, 跑完退出后平台自动释放机器, 不能 nohup 后台化
# 一次申请内顺序跑完所有多机模型: DeepSeek-R1-0528-W8A8 / GLM-5.2-W8A8 / Kimi-K2.6-Int4
# 每个模型: 启动 rank0 -> 等健康 -> client 推理(含 long-text) -> vllm bench -> 停止
# 模型切换: 本机 + worker 都确认停止干净后才进入下一个(worker_ready 握手)
set -uo pipefail

BASE="/sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16"
cd "$BASE" || exit 1
source ./config.sh

# 日志目录: 按日期+时分秒命名(不再用 JOB_ID 目录), 锁/标记文件放隐藏 meta 目录
export BASE_LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp16_luwu}"
mkdir -p "$BASE_LOG_DIR"
JOB_TAG="${JOB_ID:-default}"
META_DIR="$BASE_LOG_DIR/.luwu_meta"
mkdir -p "$META_DIR"

# 单实例锁(按 JOB_ID 区分, 保证平台重复拉起时锁稳定): 后启动的实例等待前一个完成
LOCK="$META_DIR/lock_master_${JOB_TAG}"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[luwu-master] 检测到已有 master 实例在运行, 等待其完成..."
  flock 9
  echo "[luwu-master] 前序实例已结束, 本实例直接退出"
  exit 0
fi

# 每次申请独立 run 目录: 按日期+时分秒命名, 并发布给 worker 保持一致
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BASE_LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$META_DIR/current_run_${JOB_TAG}"
LOG="$RUN_DIR/luwu_master.log"
exec > >(tee -a "$LOG") 2>&1

# 模型列表(顺序执行); 可通过环境变量覆盖, 默认全部
if [ -n "${MODELS:-}" ]; then
  # 兼容逗号分隔(陆吾 env_vars 注入)与空格分隔两种写法
  MODELS="${MODELS//,/ }"
  read -r -a MODELS <<<"$MODELS"
else
  MODELS=(DeepSeek-R1-0528-W8A8 GLM-5.2-W8A8 Kimi-K2.6-Int4)
fi
SHARED="$RUN_DIR/current_model"
WORKER_READY="$RUN_DIR/worker_ready"
rm -f "$SHARED" "$WORKER_READY"

echo "[luwu-master] $(date '+%F %T') run=$RUN_ID 开始多模型顺序测试: ${MODELS[*]}"
echo "[luwu-master] 平台注入环境变量:"
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
pip install -r /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/requirements.txt >/dev/null 2>&1 || true
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
