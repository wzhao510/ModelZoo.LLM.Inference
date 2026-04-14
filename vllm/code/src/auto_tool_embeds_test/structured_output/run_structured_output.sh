#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGFILE="$SCRIPT_DIR/../structured_output.log"
echo "Starting vllm serve, logging to $LOGFILE"
echo "Starting vllm serve..."
setsid vllm serve /mxstorage/pde_ai/models/llm/Qwen/Qwen2.5-7B-Instruct \
	--port 8080 \
	--distributed-executor-backend mp \
	-tp 2 \
    --trust-remote-code \
    --max-model-len 4096 > "$LOGFILE" 2>&1 &

VLLM_PID=$!
echo "vllm serve started with PID: $VLLM_PID"

echo "Waiting for service to start..."
sleep 240

echo "Running python test.py..."
python "$SCRIPT_DIR/struc_output/automatic_parsing.py"
sleep 10
python "$SCRIPT_DIR/struc_output/choice.py"
sleep 10
python "$SCRIPT_DIR/struc_output/grammar.py"
sleep 10
python "$SCRIPT_DIR/struc_output/math_solution.py"
sleep 10
python "$SCRIPT_DIR/struc_output/regex.py"
sleep 10
python "$SCRIPT_DIR/struc_output/response_format.py"
sleep 10

echo "Stopping vllm serve..."
kill -TERM -- -$VLLM_PID
sleep 5
kill -KILL -- -$VLLM_PID 2>/dev/null
echo "Done."
