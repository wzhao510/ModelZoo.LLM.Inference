#!/bin/bash
# 查看本节点 serve 进程与 rank0 健康状态
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

echo "=== 本容器 vllm serve 进程 ==="
ps -ef | grep "[v]llm serve" || echo "无"

echo "=== rank0 /health ==="
curl -fsS "http://${MASTER_ADDR}:${SERVE_PORT}/health" && echo || echo "未就绪"

echo "=== rank0 /v1/models ==="
curl -fsS "http://${MASTER_ADDR}:${SERVE_PORT}/v1/models" | head -c 600
echo
