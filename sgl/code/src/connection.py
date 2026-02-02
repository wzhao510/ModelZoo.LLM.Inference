from typing import Optional, List, Dict, Any
import zmq
import time
import threading
from enum import Enum
import dataclasses

from utils.utils import *

GPU_NUM_PER_NODE_DEFAULT = 8

@dataclasses.dataclass
class NodeInfo:
    ip: Optional[str] = ''
    interface: Optional[str] = ''
    ib_hcas: Optional[str] = ''
    socket: Optional[None] = None
    is_local: Optional[bool] = False


class Connection:
    def __init__(
        self,
        nodes_config: List[Dict],
        slave_port: int,
        local_ip: str,
        logger = None
    ) -> None:
        self.nodes_config = nodes_config
        self.nodes_info = []
        self.slave_port = slave_port
        self.sock_timeout = 10000 # 10s
        self.logger = logger
        self.task_logger = None
        self.send_recv_lock = threading.Lock()
        self.gpu_num_per_node = GPU_NUM_PER_NODE_DEFAULT
        self.local_ip = local_ip

    def connect(self) -> None:
        if self.nodes_config is None or len(self.nodes_config) == 0:
            node_info = NodeInfo(is_local = True)
            self.nodes_info.append(node_info)
            self.gpu_num_per_node, _ = get_gpu_mem_used()
            return

        local_ip = [self.local_ip] + get_all_local_ip(self.logger)
        context = zmq.Context()
        for node in self.nodes_config:
            node_info = NodeInfo(
                ip=node['ip'], interface=node['ifname'], ib_hcas=node['ib_hcas']
            )
            if node_info.ip in local_ip:
                node_info.is_local = True
                self.nodes_info.append(node_info)
                self.gpu_num_per_node, _ = get_gpu_mem_used()
                continue
            node_info.is_local = False

            self.logger.info(f"Connect IP: {node_info.ip}")
            socket = context.socket(zmq.REQ)
            socket.RCVTIMEO = self.sock_timeout
            socket.connect(f"tcp://{node_info.ip}:{self.slave_port}")

            # check connect timeout
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.GET_GPU_STATUS,
            )
            socket.send_string(f'{op_content.to_json()}')
            try:
                output_op = MsgContent.from_json(socket.recv_string())
                self.gpu_num_per_node = output_op.info[1]
                self.logger.info(f"connect to {node_info.ip} ({self.gpu_num_per_node} gpus) successfully!")
                socket.RCVTIMEO = -1
                node_info.socket = socket
                self.nodes_info.append(node_info)
            except zmq.Again:
                self.logger.info(f"## connect to {node_info.ip} failed, exit!")
                context.destroy(linger=0)
                exit(1)
        self.logger.info(self.nodes_info)
            
    def clean(self) -> None:
        op_content = MsgContent(type=MsgType.EXIT)
        for node in self.nodes_info:
            if node.socket is not None:
                self._send_slave_msg(node.socket, op_content)
                node.socket.close()
        self.nodes_info.clear()

    def run_cmd(self, node: NodeInfo, op_content: MsgContent) -> str:
        if node.is_local:
            if op_content.is_async:
                op_content.thread = threading.Thread(target=run_sys_cmd, args=(op_content,self.task_logger,))
                op_content.thread.start()
                time.sleep(2)
            else:
                run_sys_cmd(op_content, self.task_logger)
            return ""
        else:
            return self._run_slave_cmd(node, op_content)

    def stop_cmd(self, node: NodeInfo, op_content: MsgContent) -> None:
        if node.is_local:
            if op_content.handle is not None:
                self.task_logger.info(f'Stop [{op_content.cmd}]')
                kill_process_all(op_content.handle, self.task_logger)
            else:
                self.task_logger.info(f'{op_content.cmd} proc not exist!')
        else:
            self._stop_slave_cmd(node, op_content)

    def sync_slave_output_info(self, node: NodeInfo, info:str) -> None:
        op_content = MsgContent(
            id=get_next_op_id(),
            type=MsgType.SYNC_OUTPUT_INFO,
            info=info,
        )
        node.socket.send_string(op_content.to_json())
        output_info = node.socket.recv_string() 
        self.task_logger.info(f'## Recv {node.ip} sync output info')

    def check_slave_gpu_in_use(self, node: NodeInfo) -> None:
        op_content = MsgContent(
            id=get_next_op_id(),
            type=MsgType.GET_GPU_STATUS,
        )
        node.socket.send_string(op_content.to_json())
        output_op = MsgContent.from_json(node.socket.recv_string())
        self.task_logger.info(f'## Recv {node.ip} gpu status')

    def _run_slave_cmd(self, node: NodeInfo, op_content: MsgContent) -> str:
        return self._send_slave_msg(node.socket, op_content)

    def _stop_slave_cmd(self, node: NodeInfo, op_content: MsgContent) -> None:
        op_content.type = MsgType.STOP_CMD
        self._send_slave_msg(node.socket, op_content)

    def _send_slave_msg(self, sock: zmq.sugar.socket.Socket, op_content: MsgContent) -> str:
        with self.send_recv_lock:
            sock.send_string(op_content.to_json())
            message = sock.recv_string()
            if op_content.type != MsgType.CHECK_OUTPUT_FLAG:
                self.task_logger.info(f'## Recv {message}')
            return message

