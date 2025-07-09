import zmq
import sys
import argparse
from utils.utils import *


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
g_proc_map = {}

while True:
    message = zmq_socket.recv_string()
    op_content = OperationContent.from_json(message)
    if op_content.type == OperationType.GET:
    #....
