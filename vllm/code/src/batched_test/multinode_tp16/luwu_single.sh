#!/bin/bash
# 陆吾 queue 模式 - 单机任务容器启动脚本
# 说明: 前台串行执行, 跑完退出后平台自动释放机器, 不能后台化
# 直接跑 ModelZoo batched_test 单机流程: compile_env + pip + launch.py
# 模型清单: 默认 configs/models_single_checked_20260821.yaml, 可用环境变量 MODEL_CONFIG 覆盖
# 可选: LUWU_COMPILE=1 时, 编译安装 luwu_apply 源码(mcoplib+vllm_metax)后再跑
#
# 跑精度还是性能: 用环境变量 LUWU_TEST_MODE 切换(与多机 luwu_master.sh 完全一致)
#   LUWU_TEST_MODE=infer -> launch.py --infer --long-text-case ...(默认, 精度)
#   LUWU_TEST_MODE=perf  -> launch.py --perf(性能), 跑完自动汇总一次到 CSV
#   LUWU_TEST_MODE=all   -> 精度 + 性能
# 也可以直接给脚本传 --infer / --perf / --all(与 luwu_master.sh 的用法对齐)
set -uo pipefail

# 脚本自身所在目录(/sw_home/lli/ModelZoo.LLM.Inference/..., 避免新旧代码目录混用)
TP16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$TP16_DIR/.." && pwd)"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
# 测试模式: 单机默认只跑精度(与加开关之前的行为一致); 需要性能时 LUWU_TEST_MODE=perf
source "$TP16_DIR/luwu_mode.sh"
luwu_resolve_mode infer "$@"
# 单机任务日志单独放 tp8_luwu(与多机的 tp16_luwu 同级), 每次启动一个 run_<时间戳> 目录
LOG_DIR="${LUWU_LOG_DIR:-/sw_home/lli/model_test/tp8_luwu}"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR" 2>/dev/null || { RUN_DIR="/tmp/luwu_logs/$RUN_ID"; mkdir -p "$RUN_DIR"; }
# 性能汇总 CSV(跨 run/跨版本累积, 版本升级时看性能回退用); 想让单机/多机共用一张表时用 PERF_CSV 指定
PERF_CSV="${PERF_CSV:-$LOG_DIR/perf_summary.csv}"
export PERF_CSV LUWU_TEST_MODE LUWU_RUN_INFER LUWU_RUN_PERF
LOG="$RUN_DIR/luwu_single.log"
TEE=(tee -a "$LOG")
command -v stdbuf >/dev/null 2>&1 && TEE=(stdbuf -oL -eL tee -a "$LOG")
exec > >("${TEE[@]}") 2>&1

echo "[luwu-single] $(date '+%F %T') 开始单机任务(模型清单 $MODEL_CONFIG, 模式 $LUWU_TEST_MODE: 精度=$LUWU_RUN_INFER 性能=$LUWU_RUN_PERF) 日志: $LOG"
if [ "$LUWU_RUN_PERF" = "1" ]; then
  echo "[luwu-single] 性能汇总 CSV: $PERF_CSV(run=$RUN_ID)"
fi

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

# 按模式组装 launch.py 参数: 精度 --infer(含 long-text 用例), 性能 --perf
LAUNCH_ARGS=()
if [ "$LUWU_RUN_INFER" = "1" ]; then
  LAUNCH_ARGS+=(--infer --long-text-case configs/inference/long_text_case.yaml)
fi
if [ "$LUWU_RUN_PERF" = "1" ]; then
  LAUNCH_ARGS+=(--perf)
fi
python launch.py "${LAUNCH_ARGS[@]}" --work-dir "$RUN_DIR" \
  --model-config "$MODEL_CONFIG"
RC=$?
echo "[luwu-single] $(date '+%F %T') launch.py 结束 exit=$RC"

# 跑完性能就总结一次: 结果(launch.py --perf 的 JSON)追加进 PERF_CSV 并打印汇总表
if [ "$LUWU_RUN_PERF" = "1" ]; then
  bash "$TP16_DIR/perf_summary.sh" --scan-dir "$RUN_DIR" --csv "$PERF_CSV" \
    --run-id "$RUN_ID" --run-csv "$RUN_DIR/perf_summary.csv" --report \
    || echo "[luwu-single] WARN 性能汇总失败, 可手动执行: bash $TP16_DIR/perf_summary.sh --scan-dir $RUN_DIR --report"
fi
exit "$RC"
