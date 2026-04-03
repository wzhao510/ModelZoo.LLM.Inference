#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGFILE="$SCRIPT_DIR/../prompt_embeds.log"
echo "Starting vllm serve, logging to $LOGFILE"
echo "Starting vllm serve..."
vllm serve /mxstorage/pde_ai/models/llm/Llama/Llama-3.2-1B-Instruct \
    --port 8081 \
    --runner generate \
    --enable-prompt-embeds \
    --max-model-len 4096 > "$LOGFILE" 2>&1 &

VLLM_PID=$!
echo "vllm serve started with PID: $VLLM_PID"
echo "Waiting for service to start..."
sleep 60


echo "Running python test.py..."
python "$SCRIPT_DIR/prompt_embed_inference_with_openai_client.py"
sleep 10

echo "Stopping vllm serve..."
kill $VLLM_PID
echo "Done."
