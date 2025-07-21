import zmq
import sys
import argparse
import time
import threading

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

while True:
    message = zmq_socket.recv_string()
    op_content = OperationContent.from_json(message)
    if op_content.type == OperationType.GET:
        zmq_socket.send_string(f'{op_content.to_json()}')
    elif op_content.type == OperationType.RUN:
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