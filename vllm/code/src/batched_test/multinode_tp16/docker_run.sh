#!/bin/bash
# 创建 vllm-metax 0.25.0(.103) 容器, 10.13.81.57 / 10.13.81.58 各执行一次
set -euo pipefail

IMAGE="harbor.nx.mxcr.io/ai-release/maca/vllm-metax:0.25.0-maca.ai3.8.2.103-torch2.10-py310-ubuntu22.04-amd64"
NAME="vllm_025_lli_0820"

docker pull "$IMAGE"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[docker_run] 容器 $NAME 已存在, 如需重建请先执行: docker rm -f $NAME"
  exit 1
fi

docker run -itd --name "$NAME" \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --device=/dev/infiniband \
  --group-add video \
  --network=host \
  --uts=host \
  --ipc=host \
  --privileged=true \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --shm-size '32gb' \
  --ulimit memlock=-1 \
  -v /external:/external \
  -v /usr/local:/usr/local \
  -v /mxstorage:/mxstorage \
  -v /pde_ai:/pde_ai \
  -v /pde_share:/pde_share \
  -v /sw_home/lli:/sw_home/lli \
  "$IMAGE" /bin/bash

docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
