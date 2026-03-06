# 1 pd分离服务启动
使用code/pd_scripts下脚本可实现一键启动sglang pd分离场景下的server端，包括prefill节点、deocde节点以及mini_lb服务，下面对使用步骤进行说明

## 1.1 前提条件
脚本使用依赖下述条件

1. 执行脚本的节点能ssh免密连接到其它节点
2. 通过配置/etc/hosts，保证每个节点使用`hostname -i`命令获取到的ip与ssh连接该节点的ip一致

## 1.2 配置修改

### 1.2.1 generate_config.py修改
generate_config主要用于生成多机启动的环境变量配置，根据需求进行修改

```python
container_name='sglang-benchmark-server' # 测试启动容器名字
model_path='/oschina0/kychina/models/DeepSeek-R1-BF16_W8A8/vllm_quant_model/' # 模型路径
port=8000 # 在线测试发送请求的端口

# 节点ip列表
host_list = [
    "192.168.12.15", # decode node 0 
    "192.168.12.17", # decode node 1

    "192.168.12.11", # prefill node 0
    "192.168.12.13", # prefill node 1
]

num_prefill = 1 # prefill实例数量
num_decoder = 1 # decode实例数量
num_nodes_per_prefill = 2 # 每个prefill实例使用节点数量
num_nodes_per_decoder = 2 # 每个decode实例使用节点数量
prefill_pp_size = 2 # prefill使用pp并行的pp_size
```

### 1.2.2 template_docker.sh修改
template_docker.sh中是docker启动容器的命令，根据需求修改镜像、挂载目录

```shell
set -x

WORKDIR=$(dirname "$(readlink -f "$0")")
cd $WORKDIR

source configs/$(hostname -i).env

docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

DOCKER_IMAGE=pub-registry1.metax-tech.com/ai-opentest/master/maca/sglang:0.1.1-maca.ai20251011-38-torch2.6-py310-ubuntu22.04-amd64 # 镜像

if [[ $ROLE == "decoder" ]]; then
    LAUNCH_SCRIPT=template_decoder.sh
else
    LAUNCH_SCRIPT=template_prefill.sh
fi

docker run -itd --rm --name=$CONTAINER_NAME \
            --net=host \
            --uts=host \
            --ipc=host \
            --device=/dev/dri \
            --device=/dev/mxcd  \
            --device=/dev/infiniband \
            --privileged=true \
            --group-add video \
            --security-opt seccomp=unconfined \
            --security-opt apparmor=unconfined \
            --shm-size 100gb \
            --ulimit memlock=-1 \
            -d \
            -v /data/:/data/ \
            -v /oschina0/kychina/:/oschina0/kychina/ \ # 挂载目录，自行增减
            -v $WORKDIR:/workspace \ # 不要修改这个
            --workdir=/workspace \
            --runtime=runc -t $DOCKER_IMAGE /bin/bash -c "source /etc/profile && source ~/.bashrc && bash $LAUNCH_SCRIPT"
```

### 1.2.3 template_prefill.sh修改
template_prefill.sh中是prefill节点启动命令，多机相关参数无须修改，sglang使用的环境变量、切分方式、内存设置等参数根据需求进行修改

```shell
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
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
# export SGLANG_PP_LAYER_PARTITION="32,29"
export SGLANG_TORCH_PROFILER_DIR=/oschina0/kychina/yunlei/tmp_file/

PP=$PREFILL_PP_SIZE
TP=$(( $NNODES * 8 / $PP ))
MEMORY_FRACTION=0.84

env
# --dist-init-addr --nnodes --node-rank多机参数自动生成，不用修改
python3 -m sglang.launch_server \
    --model-path $MODEL \ # model_path在generate_config.py中进行指定，此处通过环境变量获取，也可直接指定
    --dist-init-addr $DIST_INIT_ADDR  \
    --host $(hostname -i) \
    --port 30000 \
    --tp $TP \
    --pp $PP \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --disable-radix-cache  \
    --attention-backend flashinfer  \
    --mem-fraction-static ${MEMORY_FRACTION} \
    --disaggregation-mode prefill  \
    --disaggregation-ib-device  $MCCL_IB_HCA \
    --trust-remote-code 2>&1 | tee logs/prefill_$(hostname -i).log
```
### 1.2.4 template_decoder.sh修改
template_decoder.sh中是decode节点启动命令，多机相关参数无须修改，sglang使用的环境变量、切分方式、内存设置等参数根据需求进行修改
```shell
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

# --dist-init-addr --nnodes --node-rank多机参数自动生成，不用修改
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

```

### 1.2.1 启动命令
修改配置文件之后，在code/pd_scripts中执行launch_servers.sh即可启动
```shell
bash launch_servers.sh
```

### 1.2.6 输出说明
启动之后会在code/pd_scripts下生成logs和configs目录
```markdown
│   ├──pd_scripts/
│   │   ├── logs 存放日志文件
│   │   │   ├── prefill_192.168.12.11.log
│   │   │   ├── prefill_192.168.12.13.log
│   │   │   ├── decoder_192.168.12.15.log
│   │   │   ├── decoder_192.168.12.17.log
│   │   │   ├── mini_lb.log
│   │   ├── configs 脚本运行生成的中间配置文件，可以不关注
```