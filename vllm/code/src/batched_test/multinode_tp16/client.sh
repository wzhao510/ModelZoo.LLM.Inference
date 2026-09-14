#!/bin/bash
# 推理客户端: 类似单机 launch.py --infer, 支持 --long-text-case
# 用法: MODEL_NAME=xxx [MASTER_ADDR=xxx] bash client.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

: "${MODEL_NAME:?请设置 MODEL_NAME}"
MODEL_RUN_DIR="${MODEL_RUN_DIR:-$BASE_LOG_DIR/$MODEL_NAME}"
mkdir -p "$MODEL_RUN_DIR"

BT="/sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test"
TEXT_CASE="${TEXT_CASE:-$BT/configs/inference/text_case.yaml}"
LONG_TEXT_CASE="${LONG_TEXT_CASE:-$BT/configs/inference/long_text_case.yaml}"

echo "[client] $(date '+%F %T') model=$MODEL_NAME url=http://${MASTER_ADDR}:${SERVE_PORT}"
echo "[client] text_case=$TEXT_CASE long_text_case=$LONG_TEXT_CASE"

pip install -q openai 2>/dev/null || true
python "$SCRIPT_DIR/client.py" \
  --host "$MASTER_ADDR" \
  --port "$SERVE_PORT" \
  --text-case "$TEXT_CASE" \
  --long-text-case "$LONG_TEXT_CASE" \
  --output-dir "$MODEL_RUN_DIR" \
  --tag "$MODEL_NAME"
