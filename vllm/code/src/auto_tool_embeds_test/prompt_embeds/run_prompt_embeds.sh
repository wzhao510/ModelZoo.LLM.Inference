#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGFILE="$SCRIPT_DIR/../prompt_embeds.log"
echo "Starting vllm serve, logging to $LOGFILE"
echo "Starting vllm serve..."
setsid vllm serve /mxstorage/pde_ai/models/llm/Llama/Llama-3.2-1B-Instruct \
    --port 8081 \
    --runner generate \
    --enable-prompt-embeds \
    --max-model-len 4096 > "$LOGFILE" 2>&1 &

VLLM_PID=$!
echo "vllm serve started with PID: $VLLM_PID"
echo "Waiting for service to start..."
PORT=8081
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "id"; then
        echo "Service ready after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Service NOT ready after ${MAX_WAIT}s, check $LOGFILE"
    kill -TERM -- -$VLLM_PID 2>/dev/null
    exit 1
fi

echo "Running python test.py..."
python "$SCRIPT_DIR/prompt_embed_inference_with_openai_client.py"
sleep 10

echo "Stopping vllm serve..."
kill -TERM -- -$VLLM_PID
sleep 5
kill -KILL -- -$VLLM_PID 2>/dev/null
echo "Done."
