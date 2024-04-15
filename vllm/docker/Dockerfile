ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PKGSRC
ARG PIPSRC
ARG BASE_WORK_DIR=/workspace
ARG USERNAME
ARG PASSWORD
ARG OS_TYPE
ARG BRANCH_LLM_INFERENCE=mx/dev
ARG GIT_URL_LLM_INFERENCE=http://${USERNAME}:${PASSWORD}@mercury.metax-tech.com:8080/PDE/AI/ModelZoo.LLM.Inference


RUN if [ $OS_TYPE = "ubuntu" ]; \ 
        then apt-get update && apt-get install -y --no-install-recommends \ 
            libnuma1 libelf-dev libmsgpackc2 \ 
            zip build-essential libopenblas-dev liblapack-dev\ 
            vim unzip wget \ 
            libnl-3-200 openssh-client git \ 
            && apt-get clean; \ 
    elif [ $OS_TYPE = "centos" ]; \ 
        then yum install -y libtool-ltdl numactl-libs util-linux \ 
             zip  vim wget openblas-devel lapack-devel \ 
             libnl3 openssh-clients git && \ 
             yum clean all; \  
    elif [ $OS_TYPE = "kylin" ]; \ 
        then yum install -y numactl-libs openblas-devel lapack-devel \ 
             zip  vim wget \ 
             libnl3 openssh-clients git && \ 
             yum clean all; \  
    fi 

RUN . /etc/profile.d/conda.sh &&\
    conda activate base &&\
    pip3 config set global.index-url http://repo.metax-tech.com/r/pypi/simple && \
    pip3 config set install.trusted-host repo.metax-tech.com && \
    pip3 install \
        openai \
        aiohttp

WORKDIR ${BASE_WORK_DIR}
RUN git clone -b ${BRANCH_LLM_INFERENCE} ${GIT_URL_LLM_INFERENCE}
WORKDIR ${BASE_WORK_DIR}/ModelZoo.LLM.Inference/vllm
