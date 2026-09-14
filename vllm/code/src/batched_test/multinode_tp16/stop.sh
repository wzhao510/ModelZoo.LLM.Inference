#!/bin/bash
# 停止本节点上指定模型的 vllm serve
# 用法: MODEL_NAME=xxx bash stop.sh      # 只停指定模型
#       bash stop.sh --all               # 停本容器内所有 vllm serve
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

if [ "${1:-}" = "--all" ]; then
  echo "[stop] 停止本容器内所有 vllm serve"
  pkill -f "vllm serve" || echo "[stop] 无匹配进程"
  sleep 2
  pkill -9 -f "vllm serve" 2>/dev/null || true
  exit 0
fi

MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
echo "[stop] 停止 ${MODEL_NAME} (${MODEL_PATH})"
pkill -f "vllm serve ${MODEL_PATH}" || echo "[stop] 未发现匹配进程"
sleep 2
pkill -9 -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
# 兜底清理: vllm worker 进程名被改写成 VLLM::Worker*, 需要一并杀掉, 避免残留占用 GPU
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
echo "[stop] 完成"
