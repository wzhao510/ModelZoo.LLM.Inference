#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGFILE="$SCRIPT_DIR/../tool_calling.log"
echo "Starting vllm serve, logging to $LOGFILE"
echo "Starting vllm serve..."
setsid vllm serve /mxstorage/pde_ai/models/llm/Llama/Llama-3.2-1B-Instruct \
	--port 8080 \
	--enable-auto-tool-choice \
	--tool-call-parser llama3_json \
	--chat-template "$SCRIPT_DIR/tool_chat_template_llama3.1_json.jinja" \
	--max-model-len 4096 > "$LOGFILE" 2>&1 &

VLLM_PID=$!
echo "vllm serve started with PID: $VLLM_PID"

echo "Waiting for service to start..."
sleep 240

echo "Running python tool_calling.py..."
python "$SCRIPT_DIR/tool_calling.py"
sleep 10
python "$SCRIPT_DIR/tool_calling.py"
sleep 10

echo "Stopping vllm serve..."
kill -TERM -- -$VLLM_PID
sleep 5
kill -KILL -- -$VLLM_PID 2>/dev/null
echo "Done."
