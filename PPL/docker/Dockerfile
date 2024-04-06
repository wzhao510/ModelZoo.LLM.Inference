############  1. 设置基础镜像和构建参数  ############
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
#ARG PPL_URL=http://minio-c500.metax-internal.com:9000/pde-ai/c500_pde_ppl.llm.serving_multibuild/build/package/master/ppl.llm.serving-20240327-907.128/mxc500-ppl.llm.serving-20240327-907.128-ubuntu18.04-x86_64.tar.xz


############  2. 设置环境变量  ############
ENV MACA_PATH=/opt/maca \
    MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin \
    LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:{MACA_PATH}/ucx/lib:${LD_LIBRARY_PATH} \
    USE_GEMM_NN=true


############  3. 更新apt并安装必要的工具和库（先备份 apt sources.list 文件，然后加入 metax 的源） ############
# RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic main restricted" > /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-updates main restricted" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic universe" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-updates universe" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic multiverse" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-updates multiverse" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \  
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-security main restricted" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-security universe" >> /etc/apt/sources.list && \ 
#     echo "deb http://nexus.sha-dc01.metax-tech.com/r/ubuntu/ bionic-security multiverse" >> /etc/apt/sources.list

# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends \ 
#     libnuma1 libelf-dev libmsgpackc2\
#     zip build-essential libopenblas-dev liblapack-dev\
#     vim unzip wget \
#     libnl-3-200 openssh-client git \
#     && apt-get clean


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


############  4. 配置 pip 并安装 Python 模块  ############
RUN . /etc/profile.d/conda.sh &&\
    conda activate base &&\
    pip3 config set global.index-url http://repo.metax-tech.com/r/pypi/simple && \
    pip3 config set install.trusted-host repo.metax-tech.com && \
    pip3 install \
        grpcio \
        grpcio-tools \
        flask \
        flask_socketio \
        requests \
        msgpack \
        waitress


############  5. 下载 ppl.llm.serving 并安装 ############
# WORKDIR ${BASE_WORK_DIR}
# RUN wget ${PPL_URL} && \
#     tar ppl*.tar.xz

# WORKDIR WORKDIR ${BASE_WORK_DIR}/ppl.llm.serving  && \ 
#     dpkg -i ai_deb/*.deb && \ 
#     conda activate base && \ 
#     pip install wheel/macaPMX*.whl


############  6. 下载 llm 推理代码 ############
WORKDIR ${BASE_WORK_DIR}
RUN git clone -b ${BRANCH_LLM_INFERENCE} ${GIT_URL_LLM_INFERENCE}
WORKDIR ${BASE_WORK_DIR}/ModelZoo.LLM.Inference/PPL

