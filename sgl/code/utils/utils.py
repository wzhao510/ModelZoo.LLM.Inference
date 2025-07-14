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


@dataclasses.dataclass
class TaskType(Enum):
    BENCH_NORMAL = "bench_normal"
    BENCH_RAMPUP = "bench_rampup"
    BENCH_SEARCH = "bench_search"
    ACC_CEVAL = "acc_ceval"
    ACC_MMLU = "acc_mmlu"


@dataclasses.dataclass
class TaskLaunchMode(Enum):
    online = "online"
    offline = "offline"


class OperationType(IntEnum):
    GET = auto()
    RUN = auto()
    STOP = auto()


@dataclasses.dataclass
class OperationContent:
    id: int = 0
    type: int = OperationType.GET
    envs: Optional[List[str]] = None
    cmd: Optional[str] = ''
    handle: Optional[None] = None
    output: Optional[str] = ''
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False
    statu: Optional[int] = 0
    is_master: Optional[bool] = False
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
    

def run_sys_cmd(cmd: str, proc: OperationContent):
    """Run |cmd| and return its output."""
    proc.handle = subprocess.Popen(cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = proc.handle.stdout.readline()
        if not line and proc.handle.poll() is not None:
            break

        # store output to memory or not
        if proc.store_output:
            proc.output += line
        if proc.print_output:
            print(line.strip())

        # check process ready
        if not proc.is_ready and proc.ready_flag:
            for flag in proc.ready_flag:
                if flag in line:
                    proc.is_ready = True
                    break
