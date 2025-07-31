# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import json
import os
import socket
import psutil
import dataclasses
from enum import Enum, IntEnum, auto
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess
import psutil
import signal
import logging
import re

SLAVE_GET_IOF = 'init_output_file'
SLAVE_GET_GIU = 'gpu_in_use'

_model_dir = Path(__file__).parents[2].resolve(strict=True)

def get_modelzoo_vllm_dir():
    return _model_dir


def get_model_config_filename(modelname):
    return _model_dir / modelname / "config.json"


def get_params(modelname):
    with open(get_model_config_filename(modelname), "r") as f:
        param_all = json.load(f)
    return param_all


def write_txt(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)


logger = logging.getLogger(__name__)

def configure_logger(prefix: str = "",log_file = None):
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        filename = log_file
    )

def create_file(filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass
    print(f"File {filename} has been created or already exists.")


def read_json(json_file):
    with open(json_file, 'r') as f:
        config_str = f.read()
    json_str = re.sub('//.*', '', config_str)
    json_str = re.sub('/\*.*?\*/', '', json_str, flags=re.S)
    config = json.loads(json_str)
    return config

def kill_process_all(process):
    """ kill all process  """
    try:
        cur_process = psutil.Process(process.pid)
        child_pid = cur_process.children(recursive=True)
        for child in child_pid:
            os.kill(child.pid, signal.SIGTERM)
    except Exception as e:
        print(f"kill child process exception {e}")
    try:
        process.terminate()
    except Exception as e:
        print(f"kill process exception {e}")

def kill_process_by_pid(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=3)  # 等待进程终止
        print(f"已杀死 PID 为 {pid} 的进程")
        logger.debug(f"已杀死 PID 为 {pid} 的进程")

    except psutil.NoSuchProcess:
        print(f"PID 为 {pid} 的进程不存在")
        logger.debug(f"PID 为 {pid} 的进程不存在")
    except psutil.TimeoutExpired:
        print(f"PID 为 {pid} 的进程无法在规定时间内终止")
        logger.debug(f"PID 为 {pid} 的进程无法在规定时间内终止")
    except Exception as e:
        print(f"发生错误：{e}")
        logger.error(f"kill_process_by_pid error: {e}")

def kill_all(kill_list):
    try:
        logger.info(f"kill all list {kill_list}")
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if not proc.info['cmdline']:
                continue
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                print(cmdline)
                pass
                for kill in kill_list:
                    if kill in cmdline and proc.pid != current_pid:
                        try:
                            kill_process_by_pid(proc.pid)
                            proc.terminate()
                            proc.kill()   
                            print(f"has killed {proc.pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        logger.info(f"kill all finish")
    except Exception as e:
        logger.error(f"kill all error: {e}")

def get_ip() -> str:
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass
    return "0.0.0.0"


def get_interface_by_ip(ip) -> str:
    addrs = psutil.net_if_addrs()
    for interface, addresses in addrs.items():
        for addr in addresses:
            if addr.address == ip:
                return interface
    return ''


class TaskType(Enum):
    benchmark = "benchmark"
    acc = "acc"
    rampup = "rampup"
    perf = "perf"
    search = "search"

class TaskAccType(Enum):
    mmlu = 'mmlu'
    ceval = 'ceval'

# @dataclasses.dataclass
class TaskLaunchMode(Enum):
    online = "online"
    offline = "offline"


global_operation_id = 0
def get_next_op_id() -> int:
    global global_operation_id
    global_operation_id += 1
    return global_operation_id


class OperationType(IntEnum):
    GET = auto()
    RUN = auto()
    STOP = auto()
    EXIT = auto()


@dataclasses.dataclass
class OperationContent:
    id: int = 0
    type: int = OperationType.GET
    cmd: Optional[str] = ''
    envs: Optional[Dict[str, Any]] = None
    is_async: Optional[bool] = False
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False
    status: Optional[int] = 0
    is_master: Optional[bool] = False
    thread: Optional[None] = None
    is_benching: Optional[bool] = False # 用于区别主节点的server/bench调用
    info: Dict[str, Any] = None  # 传输其他信息给slave用的参数
                                 # 包含 task_info:dict[str, str]传输给从节点时的task信息，用于合成日志目录
                                 #       ['model_name','task_id','task_type','task_launch_mode','task_full_name']
    printenv: Optional[bool] = False    # 是否在执行run_sys_cmd 期间打印环境信息
    # ....

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))
    

def run_sys_cmd(op_content: OperationContent):
    """Run |cmd| and return its output."""
    if op_content.is_master:
        logger.info(op_content.cmd)

    custom_env = os.environ.copy()
    if op_content.envs is not None:
        for env, vaule in op_content.envs.items():
            custom_env[env] = vaule
            print(f'ENV: {env}={vaule}')
    print(f'## Run Cmd: {op_content.cmd}')
    op_content.handle = subprocess.Popen(op_content.cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                         env=custom_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = op_content.handle.stdout.readline()
        if not line and op_content.handle.poll() is not None:
            # logger.info(f"[{op_content.cmd}] exit")
            if op_content.is_master:
                logger.info(f"master launch: [{op_content.cmd}] exit")
            else:
                logger.info(f"[{op_content.cmd}] exit")
            op_content.status = op_content.handle.poll()
            break

        # store output to memory or not
        if op_content.store_output:
            if op_content.output:
                op_content.output.append(line.strip())
            else:
                op_content.output = [line.strip()]

            if line and not line.isspace():    # 有些行为产生空行，比如Loading safetensors时
                if not op_content.is_benching:
                    logger.info(f'run sys cmd log : {line}')
                else:
                    logger.info(line)

        if op_content.print_output: # 不确定salve 是否需要判断 by ydm.
            print(line.strip())

        if not op_content.is_benching:
            # check process ready
            if not op_content.is_ready and op_content.ready_flag:
                for flag in op_content.ready_flag:
                    if flag in line:
                        op_content.is_ready = True
                        break
        else:
            # from original run_bench_cmd, but not realized yet.
            # if master_launch_server_abnormal():
            if False:
                try:
                    op_content.handle.terminate()
                except Exception as e:
                    logger.error(f"kill child process exception {e}")
                pass

def get_gpu_mem_used(envs) -> str:
    mem_use = OperationContent(
        id=get_next_op_id(),
        type=OperationType.RUN,
        envs=envs,
        cmd="mx-smi",
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(mem_use)
    mem_pos = 0
    percent_pos = 0
    used_percent = '0'
    mem_list = []
    for line in mem_use.output:
        if 'Bus-id' in line:
            mem_pos = line.find('Bus-id')
            percent_pos = line.find('GPU-Util')
            continue
        if mem_pos == 0 or percent_pos == 0:
            continue

        if 'MetaX' in line:
            percent_end = line.find('%', percent_pos)
            used_percent = line[percent_pos:percent_end]
        elif '65536 MiB' in line:
            mem_end = line.find('/', mem_pos)
            used_mem_mb = line[mem_pos:mem_end].strip()
            mem_list.append(f'{used_mem_mb}_{used_percent}')
        else:
            continue
    return ','.join(mem_list)

def get_python_proc(envs):
    op_content = OperationContent(
        id=get_next_op_id(),
        type=OperationType.RUN,
        envs=envs,
        cmd="ps -ef | grep python",
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(op_content)

def check_gpu_in_use(envs) -> str:
    get_python_proc(envs)
    mem_used = get_gpu_mem_used(envs)
    gpu_mem_used = mem_used.split(',')
    used_flag = False
    used_info = ''
    for gpu_id, mem_info in enumerate(gpu_mem_used):
        used_mem, used_per = mem_info.split('_')
        if int(used_mem) >= 1000 or int(used_per) > 0:
            used_flag = True
            logger.error(f'GPU {gpu_id} are in used({used_mem}, {used_per}), please check process!!')
            used_info += f'GPU {gpu_id} are in used({used_mem}, {used_per}), please check process!!\n'
    if used_flag:
        return used_info
    logger.info(f'check gpu in use: All GPUs are free')
    
    return 'All GPUs are free'

def printenv(envs):
    """
    主进程start_server期间调用(会每个任务log都加)
    """
    logger.info(f"environment:\n")
    op_content = OperationContent(
        id=get_next_op_id(),
        type=OperationType.RUN,
        envs=envs,
        cmd='printenv',
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(op_content)
