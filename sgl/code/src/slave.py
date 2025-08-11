import zmq
import sys
import argparse
import time
import threading

from src.output import OutputManager
from utils.utils import *

def run_slave_launch_server(op_content: OperationContent) -> OperationContent:
    thread = threading.Thread(target=run_sys_cmd, args=(op_content,))
    thread.start()
    time.sleep(2)
    return op_content

def set_envs(envs: Optional[List[str]]) -> Dict[str,str]:
    pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    type=int,
    default=20000,
    help="client port bind to recv msg"
)

raw_args = parser.parse_args(sys.argv[1:])

ip_addr = get_ip()
if ip_addr == '0.0.0.0':
    print("## unable to get local ip !")
    exit(1)

context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
listen_info = f"tcp://{ip_addr}:{raw_args.port}"
zmq_socket.bind(listen_info)
print(f'bind to {listen_info}, start recving...')
g_opcontent_map = {}

slave_output_manager = OutputManager(raw_args)

while True:
    message = zmq_socket.recv_string()
    op_content = OperationContent.from_json(message)
    if op_content.type == OperationType.GET:
        if op_content.info:
            if op_content.info.get(SLAVE_GET_IOF):
                slave_output_manager.init_slave_output_file(op_content.info, op_content.cmd)
                op_content.info[SLAVE_GET_IOF] = slave_output_manager.log_file
            elif op_content.info.get(SLAVE_GET_GIU):
                op_content.info[SLAVE_GET_GIU] = check_gpu_in_use(op_content.envs)
        zmq_socket.send_string(f'{op_content.to_json()}')
    elif op_content.type == OperationType.RUN:
        # slave_output_manager.init_slave_output_file(op_content.info, op_content.cmd)
        run_content = run_slave_launch_server(op_content)
        g_opcontent_map[op_content.cmd] = run_content.handle
        zmq_socket.send_string(f"run [{op_content.cmd}] success")

    elif op_content.type == OperationType.STOP:
        if op_content.cmd in g_opcontent_map.keys():
            print(f'kill {g_opcontent_map[op_content.cmd]}')
            kill_process_all(g_opcontent_map[op_content.cmd])
            zmq_socket.send_string(f"stop [{op_content.cmd}] success")
        else:
            zmq_socket.send_string(f'[{op_content.cmd}] proc not exist!')
    elif op_content.type == OperationType.EXIT:
        logger.info(f'EXIT kill')
        zmq_socket.send_string(f"exit success")
        break
