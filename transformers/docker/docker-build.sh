#!/bin/bash

# check version format
check_version_format() {
    local version=$1
    if [[ ! $version =~ ^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])-\d+$ ]]; then
        echo "version error, e.g. 20240415-929"
        exit 1
    fi
}

# define allowed os list
#TODO
ALLOWED_OS=(ubuntu18.04 ubuntu20.04 ubuntu22.04 centos8 centos9)

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
DEFAULT_HARBOR_REPO="mxcr.io/cimaster/maca-c500-pytorch"
DEFAULT_OS="ubuntu18.04"
DEFAULT_VERSION="20240415-929"
DEFAULT_TAG="$DEFAULT_VERSION-$DEFAULT_OS-amd64"
DEFAULT_BASE_IMAGE="$DEFAULT_HARBOR_REPO:$DEFAULT_TAG"

# init vars
BASE_IMAGE=""
TAG=""
OS=""
VERSION=""
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
            VERSION=$1
            check_version_format "$VERSION"
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
    if [ -n "$VERSION" ] && [ -n "$OS" ]; then
        BASE_IMAGE="$DEFAULT_HARBOR_REPO:$VERSION-$OS-amd64"
    else 
        BASE_IMAGE=$DEFAULT_BASE_IMAGE
    fi
fi

# define os type
if [[ $BASE_IMAGE == *ubuntu* ]]; then
        OS_TYPE="ubuntu"
    elif [[ $BASE_IMAGE == *centos* ]]; then
        OS_TYPE="centos"
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
TARGET_IMAGE="modelzoo.llm.inference:${TAG}"

# echo vars
echo "BASE_IMAGE: $BASE_IMAGE"
echo "VERSION: $VERSION"
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
