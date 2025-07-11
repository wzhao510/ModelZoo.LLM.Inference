
from typing import Optional, List, Dict, Any
import zmq
import time
import threading
from enum import Enum
import dataclasses

from utils.utils import *

@dataclasses.dataclass
class LocalCommandType(Enum):
    master_server = "server"
    client = "clint"

class Connection:
    def __init__(self, machine_info: Dict[str, Any], port:int) -> None:
        self.machine_socket = []
        self.machine_info = machine_info
        self.zmq_port = port
        self.master_proc = ProcStatus()
        self.master_cmd_map = {}
        self.slave_cmd_map = {}

    def connect(self) -> None:
        context = zmq.Context()
        for server in self.machine_info:
            print(f"IP: {server['ip']}")
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server['ip']}:{self.zmq_port}")
            self.machine_socket.append(socket)

    def clean(self) -> None:
        pass

    
    def run_local_cmd(self, command:str,cmd_type:LocalCommandType) -> None:
        if cmd_type == LocalCommandType.master_server:
            self.master_proc = ProcStatus(store_output=True, print_output=True, ready_flag=["The server is fired up and ready to roll!"],
                                        is_master=True)
            thread = threading.Thread(target=run_sys_cmd, args=(command, self.master_proc,))
            thread.start()
            time.sleep(2)
            self.master_cmd_map[command] = self.master_proc.handle

    def stop_local_cmd(self, command:str,cmd_type:LocalCommandType) -> None:
        if cmd_type == LocalCommandType.master_server:
            if command in self.master_cmd_map.keys():
                print(f'stop master launch server kill {self.master_cmd_map[command]}')
                kill_process_all(self.master_cmd_map[command])
            else:
                print(f'{command} proc not exist!')

    def run_slave_cmd(self,command:str) -> None:
        for sock in self.machine_socket:
            content = OperationContent(type=OperationType.RUN, cmd=command)
            self.sock_send(sock,content.to_json())
            self.slave_cmd_map[sock] = content.cmd

    def stop_slave_cmd(self) -> None:
        for sock, command in self.slave_cmd_map.items():
            content = OperationContent(type=OperationType.STOP, cmd=command)
            self.sock_send(sock,content.to_json())


    def sock_send(self,sock:zmq.sugar.socket.Socket,content:OperationContent) -> None:
        sock.send_string(content)
        message = sock.recv_string()
        print(f'## Recv {message}')




