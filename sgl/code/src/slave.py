import zmq
import sys
import argparse
import time
import random
import threading

from src.output import OutputManager, OutputDepends
from utils.utils import *

g_logger = None
g_opcontent_map = {}
g_log_file_path = None


def do_connect(sock, op_content: MsgContent) -> None:
    sock.send_string(f'connect success!')


def do_get_gpu_status(sock, op_content: MsgContent) -> None:
    global g_logger
    gpu_count, mem_used_info = get_gpu_mem_used(g_logger)
    op_content.info = [check_gpu_in_use(g_logger), gpu_count, mem_used_info]
    sock.send_string(f'{op_content.to_json()}')


def do_sync_output_info(sock, op_content: MsgContent) -> None:
    output_depends = OutputDepends.from_json(op_content.info)
    output_manager = OutputManager(output_depends)
    global g_logger, g_log_file_path
    g_logger = output_manager.logger
    g_log_file_path = output_manager.log_file_path
    sock.send_string(f'sync output info success!')


def do_run_cmd(sock, op_content: MsgContent) -> None:
    global g_logger, g_opcontent_map
    thread = threading.Thread(target=run_sys_cmd, args=(op_content,g_logger,))
    thread.start()
    time.sleep(2)
    op_content.thread = thread
    g_opcontent_map[op_content.cmd] = op_content
    sock.send_string(f"run [{op_content.cmd}] success")


def do_stop_cmd(sock, op_content: MsgContent) -> None:
    global g_logger, g_opcontent_map
    if op_content.cmd in g_opcontent_map.keys():
        log_msg_level(f'Stop [{op_content.cmd}]', g_logger)
        kill_process_all(g_opcontent_map[op_content.cmd].handle, g_logger)
        sock.send_string(f"stop [{op_content.cmd}] success")
        if not op_content.is_benching:
            kill_local_defunct_process(g_logger)
        del g_opcontent_map[op_content.cmd]
    else:
        sock.send_string(f'[{op_content.cmd}] proc not exist!')


def do_get_cmd_status(sock, op_content: MsgContent) -> None:
    global g_logger, g_opcontent_map
    if op_content.cmd not in g_opcontent_map.keys():
        op_content.status = -1
        log_msg_level(f"get cmd status failed, cmd[{op_content.cmd}] not exist", g_logger)
        sock.send_string(f'{op_content.to_json()}')
        return
    
    op_content.output = g_opcontent_map[op_content.cmd].output
    op_content.status = g_opcontent_map[op_content.cmd].status
    if op_content.status is not None:
        del g_opcontent_map[op_content.cmd]
    sock.send_string(f'{op_content.to_json()}')


def do_check_output_flag(sock, op_content: MsgContent) -> None:
    global g_logger, g_opcontent_map
    output_flags = op_content.info
    op_content.info = [None, None]
    if op_content.cmd in g_opcontent_map.keys() and g_opcontent_map[op_content.cmd].output is not None:
        output = "".join(g_opcontent_map[op_content.cmd].output)
        for flag_str in output_flags:
            if flag_str in output:
                op_content.info[0] = flag_str
                log_msg_level(f"******************************** check flag {flag_str} ********************************", g_logger)
                break
        op_content.info[1] = get_file_size(g_log_file_path)
    sock.send_string(f'{op_content.to_json()}')


def do_get_server_args(sock, op_content: MsgContent) -> None:
    global g_logger, g_opcontent_map
    op_content.info = [None, None]
    if op_content.cmd in g_opcontent_map.keys() and g_opcontent_map[op_content.cmd].output is not None:
        output = "".join(g_opcontent_map[op_content.cmd].output)
        op_content.info = match_server_args(output, op_content.cmd, g_logger)
    sock.send_string(f'{op_content.to_json()}')


def do_exit(sock, op_content: MsgContent) -> None:
    global g_logger
    log_msg_level(f'EXIT kill', g_logger)
    sock.send_string(f"exit success")


g_do_msg_map = {
    MsgType.CONNECT: do_connect,
    MsgType.GET_GPU_STATUS: do_get_gpu_status,
    MsgType.SYNC_OUTPUT_INFO: do_sync_output_info,
    MsgType.RUN_CMD: do_run_cmd,
    MsgType.STOP_CMD: do_stop_cmd,
    MsgType.GET_CMD_STATUS: do_get_cmd_status,
    MsgType.CHECK_OUTPUT_FLAG: do_check_output_flag,
    MsgType.GET_SERVER_ARGS: do_get_server_args,
    MsgType.EXIT: do_exit,
}

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

if len(ip_addr) > 0 and raw_args.local_ip not in ip_addr:
    print(f"## unable to get local ip {raw_args.local_ip} in {ip_addr}")
    exit(1)

context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
listen_info = f"tcp://{raw_args.local_ip}:{raw_args.port}"
zmq_socket.bind(listen_info)
print(f'bind to {listen_info}, start recving...')

while True:
    message = zmq_socket.recv_string()
    op_content = MsgContent.from_json(message)
    # log_msg_level(f'## Recv {op_content.cmd}', g_logger)
    if op_content.type in g_do_msg_map.keys():
        g_do_msg_map[op_content.type](zmq_socket, op_content)
    else:
        log_msg_level(f'unsupport message type {op_content.type}', g_logger)
        zmq_socket.send_string(f'unsupport message type {op_content.type}')
        continue

    if op_content.type == MsgType.EXIT:
        for cmd, op in g_opcontent_map.items():
            log_msg_level(f'Stop [{cmd}]', g_logger)
            kill_process_all(op.handle, g_logger)
            kill_local_defunct_process(g_logger)
        break
