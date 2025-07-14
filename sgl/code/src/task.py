
from typing import Optional, List
import time

from connection import Connection,LocalCommandType
from utils.utils import ProcStatus
import time
import threading

class BaseTask:
    def __init__(self, connection:Optional[Connection],launch_server:str,bench_serving:List[str],) -> None:
        self.connection = connection
        self.master_proc = ProcStatus()
        self.master_cmd_map = None
        self.timer = None
        self.launch_server_command = launch_server
        self.bench_serving_command = bench_serving
        #...

    def run(self):
        pass

    def start_timer(self):
        pass

    def stop_timer(self):
        pass

    def pre_start_server(self):
        pass

    def start_server(self):
        self.start_master_launch_server()
        self.start_slave_launch_server()


    def post_start_server(self):
        pass

    def stop_server(self):
        self.stop_slave_launch_server()
        self.stop_master_launch_server()

    def bench_test(self):
        pass

    def generate_result(self):
        pass

    def start_master_launch_server(self):
        self.connection.run_local_cmd(self.launch_server_command,LocalCommandType.master_server)

    def start_slave_launch_server(self):
        self.connection.run_slave_cmd(self.launch_server_command)

    def stop_master_launch_server(self):
        self.connection.stop_local_cmd(self.launch_server_command,LocalCommandType.master_server)

    def stop_slave_launch_server(self):
        try:
            self.connection.stop_slave_cmd()
        except Exception as _e:
            print(f"slave launch server error:{_e}")
        

class TaskOnline(BaseTask):
    def __init__(self, connection:Optional[Connection], launch_server:str, bench_serving:List[str]) -> None:
        super().__init__(connection,launch_server,bench_serving)

    @staticmethod
    def from_config(config):
        # parse config and return list of TaskOnline
        return 
    
    def run(self):
        self.start_server()
        self.bench_test()
        self.stop_server()


class TaskOffline(BaseTask):
    def __init__(self) -> None:
        pass

    @staticmethod
    def from_config(config):
        # parse config and return list of TaskOffline
        return 
    
    def run(self):
        self.start_server()
        self.stop_server()
