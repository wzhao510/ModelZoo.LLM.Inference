#!/bin/bash
# 轮询 rank0 的 /health 直到就绪
# 用法: bash check.sh [超时秒数, 默认 3600]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

TIMEOUT="${1:-3600}"
URL="http://${MASTER_ADDR}:${SERVE_PORT}/health"
LOG_HINT_DIR="${MODEL_RUN_DIR:-$BASE_LOG_DIR/$MODEL_NAME}"

echo "[check] 等待 ${URL} 就绪 (超时 ${TIMEOUT}s)"
for ((i = 1; i <= TIMEOUT; i++)); do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "[check] READY (${i}s)"
    exit 0
  fi
  if ((i % 60 == 0)); then
    echo "[check] ... 已等待 ${i}s, 可查看日志: $LOG_HINT_DIR/rank*_serve.log"
  fi
  sleep 1
done

echo "[check] TIMEOUT: ${URL} 未就绪" >&2
exit 1
