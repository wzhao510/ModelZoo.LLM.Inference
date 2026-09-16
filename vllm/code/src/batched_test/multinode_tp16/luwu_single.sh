#!/bin/bash
# 陆吾 queue 模式 - 单机任务容器启动脚本
# 说明: 前台串行执行, 跑完退出后平台自动释放机器, 不能后台化
# 直接跑 ModelZoo batched_test 单机流程: compile_env + pip + launch.py --infer --long-text-case
# 模型清单: 默认 configs/models_single_checked_20260821.yaml, 可用环境变量 MODEL_CONFIG 覆盖
# 可选: LUWU_COMPILE=1 时, 编译安装 luwu_apply 源码(mcoplib+vllm_metax)后再跑
set -uo pipefail

# 脚本自身所在目录(/sw_home/lli/ModelZoo.LLM.Inference/..., 避免新旧代码目录混用)
TP16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$TP16_DIR/.." && pwd)"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
# 单机任务日志单独放 tp8_luwu(与多机的 tp16_luwu 同级), 每次启动一个 run_<时间戳> 目录
LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp8_luwu}"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR" 2>/dev/null || { RUN_DIR="/tmp/luwu_logs/$RUN_ID"; mkdir -p "$RUN_DIR"; }
LOG="$RUN_DIR/luwu_single.log"
TEE=(tee -a "$LOG")
command -v stdbuf >/dev/null 2>&1 && TEE=(stdbuf -oL -eL tee -a "$LOG")
exec > >("${TEE[@]}") 2>&1

echo "[luwu-single] $(date '+%F %T') 开始单机任务(模型清单 $MODEL_CONFIG) 日志: $LOG"

# 可选: LUWU_COMPILE=1 时, 编译安装 luwu_apply 源码(mcoplib+vllm_metax)后再跑
if [ "${LUWU_COMPILE:-0}" = "1" ]; then
  echo "[luwu-single] LUWU_COMPILE=1, 执行源码编译安装..."
  if ! bash "$TP16_DIR/luwu_compile.sh"; then
    echo "[luwu-single] ERROR 编译安装失败, 终止"
    exit 1
  fi
fi

cd "$BASE" || exit 1
bash /sw_home/lli/compile_env.sh
pip install -r "$BASE/requirements.txt"
python launch.py --infer --work-dir "$RUN_DIR" \
  --long-text-case configs/inference/long_text_case.yaml \
  --model-config "$MODEL_CONFIG"
RC=$?
echo "[luwu-single] $(date '+%F %T') launch.py 结束 exit=$RC"
exit "$RC"
