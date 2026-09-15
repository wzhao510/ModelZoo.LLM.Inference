#!/bin/bash
# 容器内初始化: compile_env + batched_test 测试依赖
# 用法: docker exec vllm_025_lli_0820 bash /sw_home/lli/ModelZoo.LLM.Inference/vllm/code/src/batched_test/multinode_tp16/setup.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELZOO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"   # .../ModelZoo.LLM.Inference

echo "[setup] 执行 compile_env.sh ..."
bash /sw_home/lli/compile_env.sh || {
  echo "[setup] compile_env.sh 失败" >&2
  exit 1
}

echo "[setup] 安装 batched_test requirements ..."
pip install -r "$MODELZOO_ROOT/vllm/code/src/batched_test/requirements.txt" || {
  echo "[setup] requirements 安装失败" >&2
  exit 1
}

echo "[setup] 完成: $(which vllm) $(vllm --version 2>&1 | tail -1)"
