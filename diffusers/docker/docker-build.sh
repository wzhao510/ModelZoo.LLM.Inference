#!/bin/bash

# check maca version format
check_version_format() {
    local version=$1
    if [[ ! $version =~ ^[0-9]+(\.[0-9]+){3,4}$ ]]; then
        echo "version error, e.g. 2.18.0.8 or 2.19.2.26.116"
        exit 1
    fi
}

# define allowed os list
ALLOWED_OS=(ubuntu18.04 ubuntu20.04 ubuntu22.04 centos8 centos9 kylinv10 kylin2309a)

# check os
check_os() {
    local os_to_check=$1
    for os in "${ALLOWED_OS[@]}"; do
        if [[ "$os_to_check" == "$os" ]]; then
            return 0
        fi
    done
    echo "os error. Allowed values are: ${ALLOWED_OS[*]}"
    exit 1
}
  
# default vars
DEFAULT_HARBOR_REPO="mxcr.io/ai-release/c500/onnxruntime-maca"
DEFAULT_OS="ubuntu18.04"
DEFAULT_MACA_VERSION="2.20.2.1.155"
DEFAULT_TAG="$DEFAULT_MACA_VERSION-$DEFAULT_OS-amd64"
DEFAULT_BASE_IMAGE="$DEFAULT_HARBOR_REPO:$DEFAULT_TAG"

# init vars
BASE_IMAGE=""
TAG=""
OS=""
MACA_VERSION=""
OS_TYPE=""
DOCKERFILE_PATH=""
#PKGSRC="http://nexus.sha-dc01.metax-tech.com/r/ubuntu/"
PKGSRC="http://repo.metax-tech.com/r"
PIPSRC="http://repo.metax-tech.com/r/pypi/simple"
USERNAME=""
PASSWORD=""

# parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        -t) # image tag
            shift
            TAG=$1
            BASE_IMAGE="$DEFAULT_HARBOR_REPO:$TAG"
            ;;
        -u)
            shift
            USERNAME=$1
            ;;
        -p)
            shift
            PASSWORD=$1
            ;;
        -os) # os version
            shift
            OS=$1
            check_os "$OS"
            ;;
        -v) # maca version
            shift
            MACA_VERSION=$1
            check_version_format "$MACA_VERSION"
            ;;
        -d) # dockerfile path
            shift
            DOCKERFILE_PATH=$1
            ;;
        -pkgsrc) # apt or yum source
            shift
            PKGSRC=$1
            ;;
        -pipsrc) # pip source
            shift
            PIPSRC=$1
            ;;
        *)
            echo "error: unknown parameter $1"
            exit 1
            ;;
    esac
    shift
done
  
# define base image
if [ -z "$BASE_IMAGE" ]; then
    if [ -n "$MACA_VERSION" ] && [ -n "$OS" ]; then
        BASE_IMAGE="$DEFAULT_HARBOR_REPO:$MACA_VERSION-$OS-amd64"
    else 
        BASE_IMAGE=$DEFAULT_BASE_IMAGE
    fi
fi

# define os type
if [[ $BASE_IMAGE == *ubuntu* ]]; then
        OS_TYPE="ubuntu"
    elif [[ $BASE_IMAGE == *centos* ]]; then
        OS_TYPE="centos"
    elif [[ $BASE_IMAGE == *kylin* ]]; then
        OS_TYPE="kylin"
    else
        OS_TYPE="unknown"
fi

# check dockerfile path
if [ -z $DOCKERFILE_PATH ]; then
    DOCKERFILE_PATH="./Dockerfile"
fi

# check dockerfile path
if [ -z $USERNAME ] || [ -z $PASSWORD ]; then
    echo "error: username or password."
    exit 1
fi

# set target image
TAG=$(echo $BASE_IMAGE | cut -d':' -f2)
TARGET_IMAGE="modelzoo.sd.inference:${TAG}"

# echo vars
echo "BASE_IMAGE: $BASE_IMAGE"
echo "MACA_VERSION: $MACA_VERSION"
echo "OS: $OS"
echo "OS_TYPE: $OS_TYPE"
echo "TARGET_IMAGE: $TARGET_IMAGE"
echo "PKGSRC: $PKGSRC"
echo "PIPSRC: $PIPSRC"

# print docker build command
DOCKER_BUILD_COMMAND="docker build \ 
    --build-arg BASE_IMAGE=$BASE_IMAGE \ 
    --build-arg USERNAME=$USERNAME \ 
    --build-arg PASSWORD=$PASSWORD \ 
    --build-arg OS_TYPE=$OS_TYPE \ 
    --build-arg PKGSRC=$PKGSRC \  
    --build-arg PIPSRC=$PIPSRC \ 
    -t $TARGET_IMAGE \ 
    -f $DOCKERFILE_PATH \ 
    . "
echo "$DOCKER_BUILD_COMMAND"

echo "Start building docker image ......"

# build docker image
docker build \
    --no-cache \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg USERNAME=$USERNAME \
    --build-arg PASSWORD=$PASSWORD \
    --build-arg OS_TYPE=$OS_TYPE \
    --build-arg PKGSRC=$PKGSRC \
    --build-arg PIPSRC=$PIPSRC \
    -t $TARGET_IMAGE \
    -f $DOCKERFILE_PATH \
    .




