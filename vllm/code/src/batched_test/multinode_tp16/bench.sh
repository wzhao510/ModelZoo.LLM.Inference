#!/bin/bash
# 对已启动的 rank0 服务执行在线吞吐压测 (vllm bench serve 连接已有服务)
# 用法: MODEL_NAME=xxx bash bench.sh
# 默认压测点只有一个(与单机 LUWU_TEST_MODE 的压测口径一致):
#   batch(并发) 8, input 3072, output 1024  —— 便于版本升级时逐版本对比
# 可选: NUM_PROMPTS MAX_CONCURRENCY INPUT_LEN OUTPUT_LEN RESULT_DIR
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh" || exit 1

MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
INPUT_LEN="${INPUT_LEN:-3072}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
RESULT_DIR="${RESULT_DIR:-${MODEL_RUN_DIR:-$BASE_LOG_DIR/$MODEL_NAME}}"

mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/bench_in${INPUT_LEN}_out${OUTPUT_LEN}_c${MAX_CONCURRENCY}.log"
BASE_URL="http://${MASTER_ADDR}:${SERVE_PORT}"

echo "[bench] $(date '+%F %T') model=$MODEL_NAME base_url=$BASE_URL"
echo "[bench] prompts=$NUM_PROMPTS concurrency=$MAX_CONCURRENCY in=$INPUT_LEN out=$OUTPUT_LEN"

vllm bench serve \
  --base-url "$BASE_URL" \
  --trust-remote-code \
  --dataset-name random \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --temperature 0 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  2>&1 | tee "$LOG"
