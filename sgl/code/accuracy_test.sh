#!/bin/bash

CURDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

mkdir -p $CURDIR/../acc_logs
model=${1%?}
echo "all vars are $@"

if [ $2 == "triton"  ]; then
  server_param="--model ${1} --trust-remote-code --tp 1 --attention-backend triton"
  client_param="--model ${1} --test_jsonl /pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl --batch_size 32 --random_seed 0 --random_num 200 --save_dir acc_results/triton"
elif [ $2 == "flashinfer" ]; then
  server_param="--model ${1} --trust-remote-code --tp 1 --attention-backend flashinfer --enable-flashinfer-mla"
  client_param="--model ${1} --test_jsonl /pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl --batch_size 32 --random_seed 0 --random_num 200 --save_dir acc_results/flashinfer"
fi

echo "server params are ${server_param}"
echo "client params are ${client_param}"
server_log_file=$CURDIR/../acc_logs/${model##*/}_sglang_server_${2}.log
echo "server log file is ${server_log_file}"


echo "start server process..."
python -m sglang.launch_server $server_param > $server_log_file 2>&1 &
server_pid=$!

cleanup() {
  echo "terminating server process..."
  kill -9 $server_pid 2>/dev/null
  pkill -9 sglang
}
trap cleanup EXIT


echo "waiting for server process ready..."

timeout 900 tail -F $server_log_file | grep -qi "The server is fired up and ready to roll!"
if [ $? -ne 0 ]; then
    echo $?
    echo "process exceeds the time-out threshold!"
    exit 1
fi

echo "server is ready, start client test..."
client_log_file=$CURDIR/../acc_logs/${model##*/}_sglang_client_${2}.log

python ./code/run_ceval_client.py $client_param 2>&1 | tee  $client_log_file
client_exit_status=$?

exit $client_exit_status
