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
            node_info.socket = context.socket(zmq.REQ)
            node_info.socket.RCVTIMEO = self.sock_timeout
            node_info.socket.connect(f"tcp://{node_info.ip}:{self.slave_port}")

            # check connect timeout
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.GET_GPU_STATUS,
            )
            try:
                output_msg = self.send_slave_msg(node_info, op_content, self.logger)
                output_op = MsgContent.from_json(output_msg)
                self.gpu_num_per_node = output_op.info[1]
                self.logger.info(f"connect to {node_info.ip} ({self.gpu_num_per_node} gpus) successfully!")
                node_info.socket.RCVTIMEO = -1
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
                self.send_slave_msg(node, op_content, self.logger)
                node.socket.close()
        self.nodes_info.clear()

    def run_cmd(self, node: NodeInfo, op_content: MsgContent, task_logger, task_wake_event):
        if node.is_local:
            if op_content.is_async:
                op_content.thread = threading.Thread(target=run_sys_cmd, args=(op_content,task_logger,))
                op_content.thread.start()
                time.sleep(2)
            else:
                run_sys_cmd(op_content, task_logger)
        else:
            self._run_slave_cmd(node, op_content, task_logger)
            while not op_content.is_async:
                get_status_op = MsgContent(
                    id=get_next_op_id(),
                    type=MsgType.GET_CMD_STATUS,
                    cmd=op_content.cmd,
                )
                message = self.send_slave_msg(node, get_status_op, task_logger)
                output_msg = MsgContent.from_json(message)
                if output_msg.status is not None:
                    op_content.status = output_msg.status
                    op_content.output = output_msg.output
                    break
                is_set = task_wake_event.wait(timeout=10)
                if is_set:
                    break

    def stop_cmd(self, node: NodeInfo, op_content: MsgContent, task_logger) -> None:
        if node.is_local:
            if op_content.handle is not None:
                task_logger.info(f'Stop [{op_content.cmd}]')
                kill_process_all(op_content.handle, task_logger)
            else:
                task_logger.info(f'{op_content.cmd} proc not exist!')
        else:
            self._stop_slave_cmd(node, op_content, task_logger)

    def sync_output_info(self, node: NodeInfo, info:str, task_logger) -> None:
        if node.is_local:
            return
        op_content = MsgContent(
            id=get_next_op_id(),
            type=MsgType.SYNC_OUTPUT_INFO,
            info=info,
        )
        output_info = self.send_slave_msg(node, op_content, task_logger)
        task_logger.info(f'## Recv {node.ip} sync output info')

    def check_node_gpu_in_use(self, node: NodeInfo, task_logger) -> None:
        if node.is_local:
            check_gpu_in_use(task_logger)
        else:
            op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.GET_GPU_STATUS,
            )
            output_msg = self.send_slave_msg(node, op_content, task_logger)
            output_op = MsgContent.from_json(output_msg)
            task_logger.info(f'## Recv {node.ip} gpu status {output_op.info}')

    def send_slave_msg(self, node: NodeInfo, op_content: MsgContent, task_logger) -> str:
        with self.send_recv_lock:
            node.socket.send_string(op_content.to_json())
            message = node.socket.recv_string()
            # if op_content.type not in [
            #     MsgType.CHECK_OUTPUT_FLAG,
            #     MsgType.GET_SERVER_ARGS,
            #     MsgType.GET_CMD_STATUS
            #     ]:
            #     if task_logger is None:
            #         task_logger.info(f'## Recv {message}')
            #     else:
            #         task_logger.info(f'## Recv {message}')
            return message
    
    def _run_slave_cmd(self, node: NodeInfo, op_content: MsgContent, task_logger) -> str:
        return self.send_slave_msg(node, op_content, task_logger)

    def _stop_slave_cmd(self, node: NodeInfo, op_content: MsgContent, task_logger) -> None:
        op_content.type = MsgType.STOP_CMD
        self.send_slave_msg(node, op_content, task_logger)
