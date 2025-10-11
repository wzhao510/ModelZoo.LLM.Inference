#!/bin/bash

set -x
set -e

WORKDIR=$(dirname "$(readlink -f "$0")")
cd $WORKDIR

rm -rf $WORKDIR/configs/*
/usr/bin/python3 generate_config.py

rm -rf $WORKDIR/logs/*
mkdir -p $WORKDIR/logs
chmod 777 -R $WORKDIR/logs

bash configs/stop_all.sh
bash configs/launch_all.sh
bash configs/mini_lb.sh $WORKDIR/logs