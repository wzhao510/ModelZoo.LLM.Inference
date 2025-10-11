#!/bin/bash

set -x
pidof python | xargs -n 1 kill -9
source configs/$(hostname -i).env

export MCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export GLOO_SOCKET_IFNAME=inbond1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MACA_DIRECT_DISPATCH=1
export MACA_GRAPH_LAUNCH_QUEUE_POLICY=3
export MCDBG_GRAPH_LAUNCH_QUEUE_POLICY=3
# export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
# export MACA_QUEUE_SCHEDULE_POLICY=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export CUDA_GRAPH_DP_USE_SUM_BS=0
export SGLANG_TORCH_PROFILER_DIR=/oschina0/kychina/yunlei/tmp_file/


TP=16
DP=4
MEMORY_FRACTION=0.86

env


python3 -m sglang.launch_server \
    --model-path $MODEL \
    --dist-init-addr $DIST_INIT_ADDR  \
    --host $(hostname -i) \
    --port 30001 \
    --tp $TP \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --disable-radix-cache  \
    --attention-backend flashinfer  \
    --quantization w8a8_int8 \
    --dp $DP \
    --enable-dp-attention \
    --mem-fraction-static ${MEMORY_FRACTION} \
    --disaggregation-mode decode  \
    --disaggregation-ib-device  $MCCL_IB_HCA \
    --trust-remote-code 2>&1 | tee logs/decoder_$(hostname -i).log

