#!/bin/bash

CURDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

mkdir -p $CURDIR/acc_logs
server_log_file=$CURDIR/acc_logs/${1##*/}_sglang_server.log
echo "server log file is ${server_log_file}"


echo "start server process..."
python -m sglang.launch_server --model $1 --trust-remote-code --tp 1 > $server_log_file 2>&1 &
server_pid=$!


cleanup() {
  echo "terminating server process..."
  kill $server_pid 2>/dev/null
}
trap cleanup EXIT


echo "waiting for server process ready..."

timeout 360 tail -F $server_log_file | grep -qi "The server is fired up and ready to roll!"
if [[ $? -ne 0 ]]; then
    echo $?
    echo "process exceeds the time-out threshold!"
fi



echo "server is ready, start client test..."
client_log_file=$CURDIR/acc_logs/${1##*/}_sglang_client.log

python ./code/run_ceval_client.py --model $1 --test_jsonl /pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl --batch_size 32 --random_seed 0 --random_num 200 2>&1 | tee  $client_log_file
client_exit_status=$?


exit $client_exit_status
