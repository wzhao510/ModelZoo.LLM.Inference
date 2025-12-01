LOG=${LOG:-"tmpBurst_${SYS_TIME}"}
MODEL_PATH=${MODEL_PATH:-"/mxstorage/pde_ai/models/llm/Qwen/Qwen3-14B/"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"9001"}
DATASET_NAME=${DATASET_NAME:-"random"}
RESULT_JSONL=${RESULT_JSONL:-"result_sla.jsonl"}
DATASET_PATH=${DATASET_PATH:-"250910_BurstGPT.csv"}
NUM_PROMPT=${NUM_PROMPT:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-32}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-128}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}


COMMAND="vllm bench serve \
    --model ${MODEL_PATH} \
    --host ${HOST} \
    --port ${PORT} \
    --ignore-eos \
    --dataset-name ${DATASET_NAME} \
    --num-prompt ${NUM_PROMPT} \
    --append-result \
    --save-result \
    --result-filename ${RESULT_JSONL} \
    --max-concurrency ${MAX_CONCURRENCY}"

if [ "$DATASET_NAME" = "random" ]; then
    COMMAND="$COMMAND \
        --random-input-len ${RANDOM_INPUT_LEN} \
        --random-output-len ${RANDOM_OUTPUT_LEN}"
fi

if [ "$DATASET_NAME" = "burstgpt" ]; then
    COMMAND="$COMMAND \
        --dataset-path ${DATASET_PATH}"
fi

eval "$COMMAND 2>&1 | tee -a ${LOG}.log"