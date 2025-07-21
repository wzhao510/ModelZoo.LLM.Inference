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


@dataclasses.dataclass
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
            break

        # store output to memory or not
        if op_content.store_output:
            if op_content.output:
                op_content.output.append(line.strip())
            else:
                op_content.output = [line.strip()]
            print(line.strip())

        # check process ready
        if not op_content.is_ready and op_content.ready_flag:
            for flag in op_content.ready_flag:
                if flag in line:
                    op_content.is_ready = True
                    break
