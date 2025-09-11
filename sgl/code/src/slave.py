import zmq
import sys
import argparse
import time
import random
import threading

from src.output import OutputManager
from utils.utils import *

def run_slave_launch_server(op_content: OperationContent, logger) -> OperationContent:
    thread = threading.Thread(target=run_sys_cmd, args=(op_content,logger,))
    thread.start()
    time.sleep(2)
    op_content.thread = thread
    return op_content

def init_slave_output_file(info:dict, slave_cmd:str) -> str:
    """
    slave 使用上面的task_info对齐master
    """
    task_info = info['task_info']
    now = info['now']
    node_rank_match = re.search(r"--node-rank\s+(\d+)", slave_cmd)
    node_id = node_rank_match.group(1) if node_rank_match else random.randint(10, 999)

    log_file_path = os.path.join(info['output_path'], 
                            f"{now}",
                            f"{task_info['model_name']}",
                            task_info['task_type'],
                            'logs')
    log_file_name = f"{task_info['task_full_name']}_node{node_id}.log"
    log_file = os.path.join(log_file_path, log_file_name)
    if not os.path.exists(log_file):
        create_file(log_file)
    logger = get_logger(log_file_path, log_file_name)
    return log_file, logger


ip_addr = get_all_local_ip()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    type=int,
    default=20000,
    help="client port bind to recv msg"
)
parser.add_argument("--local-ip", type=str, required=True, help="default local ip")

raw_args = parser.parse_args(sys.argv[1:])

if raw_args.local_ip not in ip_addr:
    print("## unable to get local ip !")
    exit(1)

context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
listen_info = f"tcp://{raw_args.local_ip}:{raw_args.port}"
zmq_socket.bind(listen_info)
print(f'bind to {listen_info}, start recving...')
g_opcontent_map = {}
g_logger = None

while True:
    message = zmq_socket.recv_string()
    op_content = OperationContent.from_json(message)
    if op_content.type == OperationType.GET:
        if op_content.info:
            if op_content.info.get(SLAVE_GET_IOF):
                op_content.info[SLAVE_GET_IOF], g_logger = init_slave_output_file(op_content.info, op_content.cmd)
            elif op_content.info.get(SLAVE_GET_GIU):
                op_content.info[SLAVE_GET_GIU] = check_gpu_in_use(g_logger)
        zmq_socket.send_string(f'{op_content.to_json()}')
    elif op_content.type == OperationType.RUN:
        run_content = run_slave_launch_server(op_content, g_logger)
        g_opcontent_map[op_content.cmd] = run_content.handle
        zmq_socket.send_string(f"run [{op_content.cmd}] success")

    elif op_content.type == OperationType.STOP:
        if op_content.cmd in g_opcontent_map.keys():
            log_msg_level(f'Stop [{g_opcontent_map[op_content.cmd]}]', g_logger)
            kill_process_all(g_opcontent_map[op_content.cmd], g_logger)
            zmq_socket.send_string(f"stop [{op_content.cmd}] success")
            kill_local_defunct_process(g_logger)
        else:
            zmq_socket.send_string(f'[{op_content.cmd}] proc not exist!')
    elif op_content.type == OperationType.EXIT:
        log_msg_level(f'EXIT kill', g_logger)
        zmq_socket.send_string(f"exit success")
        break
