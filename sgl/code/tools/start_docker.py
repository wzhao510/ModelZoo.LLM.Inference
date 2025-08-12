import argparse
import sys
import os
import dataclasses
import subprocess
from typing import Optional, List, Dict
import threading
import json
import logging
import psutil
import time
from utils.utils import *

logger = logging.getLogger(__name__)
@dataclasses.dataclass
class ProcStatus:
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False

@dataclasses.dataclass
class SSHInfo:
    ip: Optional[str] = ''
    user: Optional[str] = ''
    passwd: Optional[str] = '' 


@dataclasses.dataclass
class DockerPsInfo:
    container_id: Optional[str] = ''
    image: Optional[str] = ''
    command: Optional[str] = ''
    created: Optional[str] = ''
    status: Optional[str] = ''
    port: Optional[str] = ''
    name: Optional[str] = ''

@dataclasses.dataclass
class DockerImagesInfo:
    repository: Optional[str] = ''
    tag: Optional[str] = ''
    image_id: Optional[str] = ''
    created: Optional[str] = ''
    size: Optional[str] = ''
    
op_list = []

def pull_docker_image(ssh_info_list,pull_args,image_repository,image_tag):
    '''
        根据registory:tag获取镜像id,如果没有找到就docker pull拉取镜像。如果镜像拉取失败退出程序。
        拉取镜像成功，然后再依次获取镜像id
    '''
    pull_flag = [0]*len(ssh_info_list)
    threads = []
    for rank, ssh_info in enumerate(ssh_info_list):
        print(f'#################################################################### pull docker images on {ssh_info.ip} ####################################################################')
        image_id = get_image_id(pull_args,ssh_info,image_repository,image_tag)
        if image_id != '':
            pull_flag[rank] = 1
            continue
        t = threading.Thread(target=run_pull_command,args=(rank,pull_flag,ssh_info,pull_args,image_repository,image_tag,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    pull_result = True
    for rank,flag in enumerate(pull_flag):
        if flag == 0:
            print(f"pull docker images fail on {ssh_info_list[rank].ip}!!!!")
            pull_result = False
    if not pull_result:
        sys.exit(-1)
    image_id = get_image_id(pull_args,ssh_info,image_repository,image_tag)
    return image_id

def launch_docker(ssh_info_list, args, iamge_id):
    '''
        获取所有容器，查看指定的容器是否存在
            不存在：创建容器
            存在:
                1.镜像id相同，并且force_rm = False,直接start容器
                2.镜像id不同或者force_rm = True，停止容器，删除容器，重新创建
    '''
    for ssh_info in ssh_info_list: 
        print(f'#################################################################### start docker on {ssh_info.ip} ####################################################################')
        start_docker(args, iamge_id, ssh_info, prepare_docker_cmds=args.prepare_docker_cmds, force_rm = args.rm_exist_docker)


def start_docker(args, image_id: str, remote_ssh_info: SSHInfo, prepare_docker_cmds: List, force_rm: bool=False):
    container_name = args.container_name
    if check_docker_exist( container_name, image_id, remote_ssh_info, force_rm):
        return
    
    print(f'## start run {container_name=} {image_id=} on {remote_ssh_info.ip }!!')
    docker_v = args.docker_v
    docker_v_cmd = ""
    for v in docker_v:
        docker_v_cmd += f"-v {v.strip()} "
    
    docker_start_cmd = f"docker run -it --net=host --uts=host --ipc=host --device=/dev/dri --device=/dev/mxcd  --device=/dev/infiniband --privileged=true \
                         --group-add video --security-opt seccomp=unconfined --security-opt apparmor=unconfined --shm-size 100gb --ulimit memlock=-1 \
                         -d \
                         --name {container_name}  \
                         {docker_v_cmd} \
                         {image_id}  \
                         /bin/bash"
    run_cmd(docker_start_cmd, remote_ssh_info)

    if not check_docker_alive(container_name, image_id, remote_ssh_info):
        print(f'## start docker {container_name=} {image_id=} on {remote_ssh_info.ip} faild !!')
        return
    

    for cmd in prepare_docker_cmds:
        if not cmd:
            continue
        full_cmd = f'docker exec {container_name} /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh;conda activate base;{cmd}"'
        run_cmd(full_cmd, remote_ssh_info)
        
def check_docker_alive( docker_name: str, image_id: str, remote_ssh_info: SSHInfo) -> bool:
    dockers = get_docker_ps(remote_ssh_info)
    if docker_name not in dockers.keys():
        return False
    same_docker = dockers[docker_name]
    is_exit = 'Exited' in same_docker.status
    if same_docker.image != image_id:
        return False
    return not is_exit

def check_docker_exist(docker_name: str, image_id: str, remote_ssh_info: SSHInfo, force_rm: bool=False) -> bool:
    dockers = get_docker_ps( remote_ssh_info)
    if docker_name not in dockers.keys():
        return False
    same_docker = dockers[docker_name]
    is_exit = 'Exited' in same_docker.status
    if same_docker.image == image_id and not force_rm:
        print(f'## check {docker_name} {image_id} exist, no need create new!!')
        if is_exit:
            start_cmd = f'docker start {docker_name}'
            run_cmd(start_cmd,  remote_ssh_info)
        return True
    else:
        if force_rm:
            print(f'## check {docker_name} exist, but force rm , stop and rm {docker_name}!!')
        else:
            print(f'## check {docker_name} exist, but image id not equal(new:{image_id}, old:{same_docker.image} , stop and rm {docker_name}!!')
        if not is_exit:
            stop_cmd = f'docker stop {docker_name}'
            run_cmd(stop_cmd, remote_ssh_info)
        rm_cmd = f'docker rm {docker_name}'
        run_cmd(rm_cmd,  remote_ssh_info)
        return False

def exec_docker(ssh_info_list,args):
    # exec_cmd = f"docker exec -it {args.container_name}  /bin/bash'"
    exec_cmd = f"docker exec -it {args.container_name}  /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh; conda activate base'"
    for ssh_info in ssh_info_list:
        run_cmd(exec_cmd,False,ssh_info)
    
def get_docker_ps( remote_ssh_info: SSHInfo) -> Dict[str, DockerPsInfo]:
    ps_cmd = 'docker ps -a'
    proc_status = run_cmd(ps_cmd, remote_ssh_info)
    docker_ps_name = {}
    split_pos = [0]
    begin = False
    for line in proc_status.output:
        if 'CONTAINER ID' in line:
            split_pos.append(line.find('IMAGE'))
            split_pos.append(line.find('COMMAND'))
            split_pos.append(line.find('CREATED'))
            split_pos.append(line.find('STATUS'))
            split_pos.append(line.find('PORTS'))
            split_pos.append(line.find('NAMES'))
            begin = True
            continue
        if not begin:
            continue
        if len(line) < split_pos[-1]:
            break
        attr_list = []
        for i, _ in enumerate(split_pos):
            if i + 1 == len(split_pos):
                attr_list.append(line[split_pos[i]:].strip())
            else:
                attr_list.append(line[split_pos[i]:split_pos[i + 1]].strip())

        if len(attr_list) != len(dataclasses.fields(DockerPsInfo)):
            print(f'## invalid docker ps line: {line}')
            break
        docker_info = DockerPsInfo(*attr_list)
        docker_ps_name[docker_info.name] = docker_info
    return docker_ps_name


    

def get_docker_images(remote_ssh_info: SSHInfo) -> Dict[str, DockerImagesInfo]:
    images_cmd = 'docker images'
    proc_status = run_cmd(images_cmd,  remote_ssh_info)
    docker_images = {}
    split_pos = [0]
    begin = False
    for line in proc_status.output:
        if 'REPOSITORY' in line:
            split_pos.append(line.find('TAG'))
            split_pos.append(line.find('IMAGE ID'))
            split_pos.append(line.find('CREATED'))
            split_pos.append(line.find('SIZE'))
            begin = True
            continue
        if not begin:
            continue
        if len(line) < split_pos[-1]:
            break
        attr_list = []
        for i, _ in enumerate(split_pos):
            if i + 1 == len(split_pos):
                attr_list.append(line[split_pos[i]:].strip())
            else:
                attr_list.append(line[split_pos[i]:split_pos[i + 1]].strip())

        if len(attr_list) != len(dataclasses.fields(DockerImagesInfo)):
            print(f'## invalid docker ps line: {line}')
            break
        image_info = DockerImagesInfo(*attr_list)
        docker_images[f"{image_info.repository}:{image_info.tag}"] = image_info
    return docker_images

def stop_docker_container(ssh_info_list,container_name):
    '''
        停止所有设备的容器
    '''
    print("stop_docker_container#########################")
    docker_cmd = f"docker stop {container_name}"
    for ssh_info in ssh_info_list:
        run_cmd(docker_cmd,ssh_info)
    
def get_image_id(args,remote_ssh_info: SSHInfo,image_repository,image_tag):
    images = get_docker_images(remote_ssh_info)
    if f'{image_repository}:{image_tag}' not in images:
        return ''
    image_id = images[f'{image_repository}:{image_tag}'].image_id
    return image_id

def run_pull_command(rank,pull_flag,ssh_info,args,image_repository,image_tag):
    pull_cmd = f"docker pull {image_repository}:{image_tag}"
    proc = run_cmd(pull_cmd, ssh_info)
    if f"Status: Downloaded newer image for {image_repository}:{image_tag}" in proc.output or \
       f"Status: Image is up to date for {image_repository}:{image_tag}" in proc.output:
       pull_flag[rank] = 1



def get_ssh_info_list(args,json_file_path):
    '''
        获取服务器信息列表
    '''
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    ssh_info_list = []
    machine_info_list = data['machine_info']
    for  machine_info in machine_info_list:       
        ssh_info_list.append(SSHInfo( machine_info['ip'],args.user))
    return ssh_info_list

def run_cmd(cmd: str, remote_ssh_info: SSHInfo,use_thread = False) -> ProcStatus:
    """"
        is_master:是否使用ssh命令,本机不需要使用
    """
    local_ip = get_ip()
    if local_ip == '0.0.0.0':
        print("## unable to get local ip !")
        exit(1)
    if remote_ssh_info.ip == local_ip:
        is_master = True
    else:
        is_master = False
        cmd =  f"ssh -o StrictHostKeyChecking=no {remote_ssh_info.user}@{remote_ssh_info.ip} '{cmd}'"

    op_content = OperationContent(
        id=get_next_op_id(),
        type=OperationType.RUN,
        envs=None,
        cmd=cmd,
        store_output=True,
        is_ready=True,
        is_async=True,
        is_master=is_master)
    if use_thread:
        op_content.thread = threading.Thread(target = run_sys_cmd,args =(op_content,))
        op_content.thread.start()
        op_list.append(op_content)
    else:
        run_sys_cmd(op_content)
    return op_content

def start_models(args,ssh_info_list,tag):
    slave_cmd = f'docker exec -i {args.container_name} /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh; conda activate base;cd {args.target_path}/code;python3 -m src.slave --port {args.port}"'
    incremental = "--incremental-mode" if args.incremental_mode else ""
    specify = "--specify-task" if args.specify_task else ""
    benchmark_cmd = f'docker exec -i {args.container_name}  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh; conda activate base;cd {args.target_path}/code; \
        python3 -u -m src.master \
            --output-path {args.output_path} \
            --image-tag {tag} \
            --task {" ".join(args.tasks)} \
            --port {args.port} \
            {incremental} \
            {specify}"'
    master_ssh_info = None
    # 从服务器启动slave
    for ssh_info in ssh_info_list:
        local_ip = get_ip()
        if local_ip == '0.0.0.0':
            print("## unable to get local ip !")
            exit(1)
        if ssh_info.ip != local_ip:
            run_cmd(slave_cmd,ssh_info,use_thread = True)
        else:
            master_ssh_info = ssh_info
    # 主服务器执行master脚本
    time.sleep(3)
    run_cmd(benchmark_cmd,master_ssh_info,use_thread = True)
           
def get_args():
    parser = argparse.ArgumentParser()
    # 容器相关参数
    parser.add_argument("--container-name", type=str, default="dockertest_dyl", help="container name")
    parser.add_argument("--container-config", type=str,nargs='*', default=[], help="eg:--docker-config  repository1:tag    repository2:tag   repository1:tag   repository2:tag")
    parser.add_argument("--container-cycles", type=int, default=1, help="Number of cycles")
    parser.add_argument("--docker-v", type=str,nargs='*', default=[], help="Mounting path")
    parser.add_argument("--rm-exist-docker", action="store_true", help="force remove exist docker")
    parser.add_argument("--prepare-docker-cmds",nargs='*',type=str,default=[],help="Command for initializing the environment")
    parser.add_argument("--target-path", type=str, default="/pde_ai/share/sgl_automation/", help="Target work path")
    # 模型相关参数
    parser.add_argument("--output-path", type=str, default="output", help="Path for storing results")
    parser.add_argument("--tasks", type=str,nargs='*', default=["DeepSeek-R1-BF16/benchmark.json"], help="JSON file describing the task list")
    parser.add_argument("--user", type=str, default="root", help="client user")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    parser.add_argument("--incremental-mode", action="store_true", help="only run case not in pass file")
    parser.add_argument("--specify-task", action="store_true", help="Starting from the designated task")

    args = parser.parse_args(sys.argv[1:])
    return args

def run(args,ssh_info_list,repository,tag):
    # 获取镜像id
    image_id = pull_docker_image(ssh_info_list,args,repository,tag)
    # 启动容器
    launch_docker(ssh_info_list, args, image_id)
    # 杀死其他进程,这里没有设计exit_sglang.py文件，此功能暂时不加
    # kill_list = ['sglang','python','python3']
    # kill_all(kill_list)
    # 启动模型脚本
    start_models(args,ssh_info_list,tag)


        
if __name__ == "__main__":
    args = get_args()
    ssh_info_list = get_ssh_info_list(args,args.tasks[0])
    container_list = args.container_config * args.container_cycles
    for config in container_list:
        repository, tag = config.split(":")
        run(args,ssh_info_list,repository,tag)
        # 等待每一轮测试结束
        for op in op_list:
            op.thread.join()
        op_list.clear()

    stop_docker_container(ssh_info_list,args.container_name)
     
