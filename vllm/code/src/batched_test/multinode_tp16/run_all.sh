#!/bin/bash
# rank0 侧一键流程: 等待就绪 -> 压测 -> (默认)停止
# 前置: 已在 10.13.81.58 容器内执行 RANK=1 bash start.sh
# 用法: MODEL_NAME=GLM-5.2-W8A8 bash run_all.sh
# 可选: KEEP_SERVER=1 压测后保留服务; CHECK_TIMEOUT=3600; 压测参数同 bench.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

: "${MODEL_NAME:?请设置 MODEL_NAME}"

echo "[run_all] $(date '+%F %T') model=$MODEL_NAME (rank0 侧)"

"$SCRIPT_DIR/check.sh" "${CHECK_TIMEOUT:-3600}" || {
  echo "[run_all] 服务未就绪, 请检查两台机器的 rank*_serve.log" >&2
  exit 1
}

"$SCRIPT_DIR/bench.sh"
echo "[run_all] 压测完成: $BASE_LOG_DIR/$MODEL_NAME/"

if [ "${KEEP_SERVER:-0}" != "1" ]; then
  echo "[run_all] 停止本节点 serve (KEEP_SERVER=1 可保留)"
  "$SCRIPT_DIR/stop.sh"
  echo "[run_all] 请记得在 10.13.81.58 容器内也执行 bash stop.sh"
fi
