
set -x

WORKDIR=$(dirname "$(readlink -f "$0")")
cd $WORKDIR

source configs/$(hostname -i).env

docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

DOCKER_IMAGE=pub-registry1.metax-tech.com/ai-opentest/master/maca/sglang:0.5.1-maca.ai20251011-38-torch2.6-py310-ubuntu22.04-amd64

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
            -v /oschina0/kychina/:/oschina0/kychina/ \
            -v $WORKDIR:/workspace \
            --workdir=/workspace \
            --runtime=runc -t $DOCKER_IMAGE /bin/bash -c "source /etc/profile && source ~/.bashrc && bash $LAUNCH_SCRIPT"