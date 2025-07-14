
from typing import Optional, List, Dict, Any
import time

from connection import Connection,LocalCommandType
from output import OutputManager
from utils.utils import *



class BaseTask:
    def __init__(self, connection:Optional[Connection],launch_server:str,environment_variables:Optional[Dict]) -> None:
        self.connection = connection
        self.timer = None
        self.launch_mode = None
        self.task_type = None
        self.launch_server_command = launch_server
        self.output_maneger = None
        self.envs = None
        self.set_envs(environment_variables)
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

    def wait_server_ready(self) -> bool:
        pass

    def bench_test(self):
        # run client and store out to file
        # ...
        self.output_maneger.write_client_result()
        self.output_maneger.write_pass_case()
        self.output_maneger.write_real_progress()
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
    
    def set_output_maneger(self,output_maneger:OutputManager) -> None:
        self.output_maneger = output_maneger

    def set_envs(self,environment_variables) -> None:
        pass

    def check_abnormal(self):
        pass
        

class TaskOnline(BaseTask):
    def __init__(self, connection:Optional[Connection], launch_server:str, bench_serving:List[str],environment_variables:Optional[Dict]) -> None:
        super().__init__(connection,launch_server,environment_variables)
        self.bench_serving_command = bench_serving

    @staticmethod
    def from_config(config):
        # parse config and return list of TaskOnline
        task_type = config['task_type']
        if task_type == TaskType.BENCH_NORMAL.value:
            pass
        elif task_type == TaskType.BENCH_RAMPUP.value:
            pass
        elif task_type == TaskType.BENCH_SEARCH.value:
            pass
        elif task_type == TaskType.ACC_CEVAL.value:
            pass
        elif task_type == TaskType.ACC_MMLU.value:
            pass
        return 
    
    def run(self):
        self.check_abnormal()
        self.start_server()
        if self.wait_server_ready():
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
        self.check_abnormal()
        self.start_server()
        self.stop_server()
