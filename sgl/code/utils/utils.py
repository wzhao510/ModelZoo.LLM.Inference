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
import time

SLAVE_GET_IOF = 'init_output_file'
SLAVE_GET_GIU = 'gpu_in_use'
SLAVE_GET_GC = 'gpu_count'

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

class HandlerFilter(logging.Filter):
    def __init__(self, handler_names):
        super().__init__()
        self.handler_names = handler_names
    
    def filter(self, record):
        if hasattr(record, 'exclude_handler_names'):
            return record.exclude_handler_names not in self.handler_names
        return True


def get_logger(log_path, log_name):
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    except Exception as e:
        print(f'error occurred {e}')
    logger = logging.getLogger(os.path.join(log_path, log_name))
    if len(logger.handlers) > 0:
        return logger
    logger.setLevel(logging.DEBUG)

    log_file_name = os.path.join(log_path, log_name)
    fh = logging.FileHandler(log_file_name, encoding='utf-8', mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh.setFormatter(formatter)
    fh.flush = lambda: fh.stream.flush()
    fh.addFilter(HandlerFilter(['file', 'all']))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    ch.addFilter(HandlerFilter(['console', 'all']))
    logger.addHandler(ch)

    return logger


def log_msg_level(msg, logger = None, level = logging.INFO):
    if logger:
        logger.info(msg)
    else:
        print(msg, flush=True)


def create_file(filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass
    print(f"File {filename} has been created or already exists.")


def read_json(json_file, replacements = None):
    with open(json_file, 'r') as f:
        config_str = f.read()
    json_str = re.sub(r'^[ \t]*//.*(?:\r?\n)?', '', config_str, flags=re.MULTILINE)
    json_str = re.sub('/\*.*?\*/', '', json_str, flags=re.S)

    if replacements is not None:
        # 替换 ${var_name} 格式的占位符
        def replace_var(match):
            var_name = match.group(1)
            return str(replacements[var_name] if var_name in replacements.keys() else match.group(0))
        pattern = r'\$\{([^}]+)\}'
        json_str = re.sub(pattern, replace_var, json_str)
    config = json.loads(json_str)
    return config

def kill_process_all(process, logger = None):
    """ kill all process  """
    try:
        cur_process = psutil.Process(process.pid)
        child_pid = cur_process.children(recursive=True)
        for child in child_pid:
            os.kill(child.pid, signal.SIGTERM)
    except Exception as e:
        log_msg_level(f"kill child process exception {e}", logger)
    try:
        process.terminate()
    except Exception as e:
        log_msg_level(f"kill process exception {e}", logger)


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


def get_all_local_ip(logger = None) -> str:
    addr_list = []
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        addr_list.append(host_ip)
        log_msg_level(f'found ip list: {addr_list}', logger)
        return addr_list

    addrs = psutil.net_if_addrs()
    for _, addresses in addrs.items():
        for address in addresses:
            # 检查地址类型是否为IPv4或IPv6
            if str(address.family) not in ['AddressFamily.AF_INET']:
                continue
            if address.address == '127.0.0.1' or address.address == '172.17.0.1':
                continue
            addr_list.append(address.address)
    log_msg_level(f'found ip list: {addr_list}', logger)
    return addr_list


def get_json_config_default(config, key, default_value):
    return config[key] if key in config.keys() else default_value


class BenchmarkType(Enum):
    perf = "perf"
    mmlu = "mmlu"
    ceval = 'ceval'


class TaskLaunchMode(Enum):
    online = "online"
    offline = "offline"


global_operation_id = 0
def get_next_op_id() -> int:
    global global_operation_id
    global_operation_id += 1
    return global_operation_id


class MsgType(IntEnum):
    CONNECT = auto()
    GET_GPU_STATUS = auto()
    SYNC_OUTPUT_INFO = auto()
    RUN_CMD = auto()
    STOP_CMD = auto()
    GET_CMD_STATUS = auto()
    CHECK_OUTPUT_FLAG = auto()
    GET_SERVER_ARGS = auto()
    EXIT = auto()


@dataclasses.dataclass
class MsgContent:
    id: int = 0
    type: int = MsgType.CONNECT
    cmd: Optional[str] = ''
    envs: Optional[Dict[str, Any]] = None
    is_async: Optional[bool] = False
    handle: Optional[None] = None
    output: Optional[List[str]] = None
    store_output: Optional[bool] = False
    print_output: Optional[bool] = True
    ready_flag: Optional[List[str]] = None
    is_ready: Optional[bool] = False
    status: Optional[int] = None
    is_master: Optional[bool] = False
    thread: Optional[None] = None
    is_benching: Optional[bool] = False # 用于区别主节点的server/bench调用
    info: Optional[Any] = None  # 传输其他信息给slave用的参数
    printenv: Optional[bool] = False    # 是否在执行run_sys_cmd 期间打印环境信息
    special_logger: Optional[None] = None

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
    

def run_sys_cmd(op_content: MsgContent, logger = None):
    """Run |cmd| and return its output."""
    custom_env = os.environ.copy()
    if op_content.envs is not None:
        for env, vaule in op_content.envs.items():
            custom_env[env] = vaule
            log_msg_level(f'ENV: {env}={vaule}', logger)
    log_msg_level(f'## Run Cmd: {op_content.cmd}', logger)
    if op_content.special_logger is not None:
        op_content.special_logger.info(f'## Run Cmd: {op_content.cmd}', extra={'exclude_handler_names': 'console'})
    op_content.handle = subprocess.Popen(op_content.cmd, shell=True, bufsize=1, text=True, encoding='utf-8',
                                         env=custom_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = op_content.handle.stdout.readline()
        if not line and op_content.handle.poll() is not None:
            log_msg_level(f"[{op_content.cmd}] exit", logger)
            op_content.status = op_content.handle.poll()
            break

        line_striped = line.strip()
        # store output to memory or not
        if op_content.store_output:
            if op_content.output:
                op_content.output.append(line_striped)
            else:
                op_content.output = [line_striped]

        if line and not line.isspace():    # 有些行为产生空行，比如Loading safetensors时
            log_msg_level(line_striped, logger)
            if op_content.special_logger is not None:
                op_content.special_logger.info(line_striped, extra={'exclude_handler_names': 'console'})

        if not op_content.is_ready and op_content.ready_flag:
            for flag in op_content.ready_flag:
                if flag in line:
                    op_content.is_ready = True
                    break


def get_gpu_mem_used(logger = None):
    mem_use = MsgContent(
        id=get_next_op_id(),
        type=MsgType.RUN_CMD,
        cmd="mx-smi",
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(mem_use, logger)
    mem_pos = 0
    percent_pos = 0
    used_percent = '0'
    mem_list = []
    gpu_count = 0
    for line in mem_use.output:
        if 'Attached GPUs' in line:
            gpu_count = int(line.split()[-1].strip())
            continue
        if 'Bus-id' in line:
            mem_pos = line.find('Bus-id')
            percent_pos = line.find('GPU-Util')
            continue
        if mem_pos == 0 or percent_pos == 0:
            continue

        if 'MetaX' in line:
            percent_end = line.find('%', percent_pos)
            used_percent = line[percent_pos:percent_end]
        elif ' MiB' in line:
            mem_end = line.find('/', mem_pos)
            used_mem_mb = line[mem_pos:mem_end].strip() 
            mem_list.append(f'{used_mem_mb}_{used_percent}')
        else:
            continue
    return gpu_count, ','.join(mem_list)

def get_python_proc(logger):
    op_content = MsgContent(
        id=get_next_op_id(),
        type=MsgType.RUN_CMD,
        cmd="ps -ef",
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(op_content, logger)

def check_port_in_use(logger):
    log_msg_level(f'## LISTEN PORT LISTS:', logger)
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN':
            log_msg_level(f'{conn.laddr.ip}:{conn.laddr.port} pid={conn.pid}', logger)

def check_gpu_in_use(logger) -> str:
    printenv(logger)
    check_port_in_use(logger)
    get_python_proc(logger)
    _, mem_used = get_gpu_mem_used(logger)
    gpu_mem_used = mem_used.split(',')
    used_flag = False
    used_info = ''
    for gpu_id, mem_info in enumerate(gpu_mem_used):
        used_mem, used_per = mem_info.split('_')
        if int(used_mem) > 2000 or int(used_per) > 0:
            used_flag = True
            log_msg_level(f'GPU {gpu_id} are in used({used_mem}, {used_per}), please check process!!', logger)
            used_info += f'GPU {gpu_id} are in used({used_mem}, {used_per}), please check process!!\n'
    if used_flag:
        return used_info
    log_msg_level(f'check gpu in use: All GPUs are free', logger)
    
    return 'All GPUs are free'

def printenv(logger):
    """
    主进程start_server期间调用(会每个任务log都加)
    """
    log_msg_level(f"environment:", logger)
    op_content = MsgContent(
        id=get_next_op_id(),
        type=MsgType.RUN_CMD,
        cmd='printenv',
        store_output=True,
        is_master=True,
    )
    run_sys_cmd(op_content, logger)


def kill_local_defunct_process(logger = None):
    kill_cmds = [
        "pkill -9 -f sglang::",
        "pkill -9 -f sglang.launch_server",
        "pkill -9 -f sglang.bench_serving",
        "pkill -9 -f multiprocessing."
    ]
    for kill_cmd in kill_cmds:
        time.sleep(1)
        op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.RUN_CMD,
                cmd=kill_cmd,
                store_output=True,
                is_ready=True
        )
        run_sys_cmd(op_content, logger)
    time.sleep(1)


def get_folder_size(folder_path):
    total_size = 0
    if folder_path is None:
        total_size
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


def convert_str_to_env_dict(env_strs: List[str]):
    if env_strs is None or len(env_strs) == 0:
        return {}
    env_dict = {}
    for env_str in env_strs:
        position = env_str.find('=')
        if position == -1:
            continue
        env_dict[env_str[:position]] = env_str[position+1:]
    return env_dict


def match_server_args(outputs, cmd, logger):
    match = re.search(r'server_args=ServerArgs\((.*?)\)',  outputs, re.DOTALL)
    if match:
        return [match.group(0), match.group(1)]
    else:
        log_msg_level(f"未找到 server_args=ServerArgs(...) 这一行, cmd:{cmd}", logger)
        return [None, None]