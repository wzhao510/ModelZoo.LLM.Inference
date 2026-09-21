#!/bin/bash
# rank0 侧一键流程: 等待就绪 -> 精度(client)/性能(bench) -> (默认)停止
# 前置: 已在 10.13.81.58 容器内执行 RANK=1 bash start.sh
# 用法: MODEL_NAME=GLM-5.2-W8A8 bash run_all.sh
#       MODEL_NAME=GLM-5.2-W8A8 LUWU_TEST_MODE=all bash run_all.sh    # 精度 + 性能
#       MODEL_NAME=GLM-5.2-W8A8 bash run_all.sh --infer               # 只跑精度
# 模式默认 perf(与加开关之前一致: 只压测); 取值同 luwu_master.sh, 见 luwu_mode.sh
# 可选: KEEP_SERVER=1 压测后保留服务; CHECK_TIMEOUT=3600; 压测参数同 bench.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1
# 命令行也可以直接写 --infer / --perf / --all
luwu_resolve_mode perf "$@"

: "${MODEL_NAME:?请设置 MODEL_NAME}"

echo "[run_all] $(date '+%F %T') model=$MODEL_NAME (rank0 侧, 模式 $LUWU_TEST_MODE: 精度=$LUWU_RUN_INFER 性能=$LUWU_RUN_PERF)"

"$SCRIPT_DIR/check.sh" "${CHECK_TIMEOUT:-3600}" || {
  echo "[run_all] 服务未就绪, 请检查两台机器的 rank*_serve.log" >&2
  exit 1
}

RUN_DIR="${MODEL_RUN_DIR:-$BASE_LOG_DIR/$MODEL_NAME}"

if [ "$LUWU_RUN_INFER" = "1" ]; then
  MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$RUN_DIR" "$SCRIPT_DIR/client.sh"
  echo "[run_all] 精度推理完成: $RUN_DIR/"
else
  echo "[run_all] 模式 $LUWU_TEST_MODE: 跳过精度(client.sh)"
fi

if [ "$LUWU_RUN_PERF" = "1" ]; then
  MODEL_NAME="$MODEL_NAME" MODEL_RUN_DIR="$RUN_DIR" "$SCRIPT_DIR/bench.sh"
  echo "[run_all] 压测完成: $RUN_DIR/"
  # 跑完一次性能就总结一次到 CSV(幂等), 便于版本升级时对比吞吐/时延
  luwu_summarize_perf_model "$MODEL_NAME" "$RUN_DIR" --report \
    || echo "[run_all] WARN 性能汇总失败, 可手动执行: bash $SCRIPT_DIR/perf_summary.sh --scan-dir $RUN_DIR --report"
else
  echo "[run_all] 模式 $LUWU_TEST_MODE: 跳过性能(bench.sh)"
fi

if [ "${KEEP_SERVER:-0}" != "1" ]; then
  echo "[run_all] 停止本节点 serve (KEEP_SERVER=1 可保留)"
  "$SCRIPT_DIR/stop.sh"
  echo "[run_all] 请记得在 10.13.81.58 容器内也执行 bash stop.sh"
fi
