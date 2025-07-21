from typing import Optional, List, Dict, Any
import time

from src.connection import Connection
from src.output import OutputManager
from utils.utils import *


class BaseTask:
    def __init__(self,
        connection: Optional[Connection],
        launch_server: str,
        envs: Optional[Dict]
    ) -> None:
        self.connection = connection
        self.timer = None
        self.launch_mode = None
        self.task_name = None
        self.task_type = None
        self.server_cmd = launch_server
        self.server_port = 5000
        self.output_manager = None
        self.envs = envs
        self.timeout = None
        self.timer = None
        self.stopped = False

        # todo not use all node
        self.nodes_used = self.connection.nodes_info
        self.server_cmd_ops = []

    def run(self):
        pass

    def start_timer(self):
        pass

    def stop_timer(self):
        pass

    def start_server(self):
        nodes_num = len(self.nodes_used)
        for index, node in enumerate(self.nodes_used):
            cmd = self.server_cmd
            if nodes_num > 1:
                cmd += f' --dist-init-addr {self.nodes_used[0].ip}:{self.server_port}'
                cmd += f'  --nnodes {nodes_num} --node-rank {index}'

            op_content = OperationContent(
                id=get_next_op_id(),
                type=OperationType.RUN,
                envs=self.envs,
                cmd=cmd,
                store_output=True,
                is_ready=True,
                is_async=True,
            )
            self.connection.run_cmd(node, op_content)
            self.server_cmd_ops.append(op_content)

    def stop_server(self):
        self.stop_timer()
        for node, op in zip(self.nodes_used, self.server_cmd_ops):
            self.connection.stop_cmd(node, op)
        self.stopped = True

    def wait_server_ready(self) -> bool:
        pass

    def bench_test(self):
        # run client and store out to file
        # ...
        self.output_manager.write_client_result()
        self.output_manager.write_pass_case()
        self.output_manager.write_real_progress()
        pass

    def generate_result(self):
        pass
    
    def set_output_manager(self, output_manager: OutputManager) -> None:
        self.output_manager = output_manager

    def set_envs(self, environment_variables: Optional[Dict]) -> None:
        global g_env
        g_env = os.environ.copy()

        # reset slave env
        self.connection.run_slave_cmd(f'ENVRESET@ENVRESET')

        # set master and slave env
        for key, value in environment_variables.items():
            # master env
            g_env[key] = value

            # slave env
            self.connection.run_slave_cmd(f"{key}={value}")

        # TODO 
        # content obj Additional parameters are needed in the future
        content = OperationContent(cmd="printenv", cmd_type=LocalCommandType.sync_run)
        self.connection.run_local_cmd(content)
        # test log, delete later
        print('*****************set env end*************************')

    def check_abnormal(self):
        pass


class TaskOnline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        bench_serving: List[str],
        envs: Optional[Dict],
    ) -> None:
        super().__init__(connection, launch_server, envs)
        self.bench_serving = bench_serving
        self.current_bench_op = None
    
    def run(self):
        self.check_abnormal()
        self.start_server()
        if self.wait_server_ready():
            self.bench_test()
        self.stop_server()

    def wait_server_ready(self):
        while True:
            if self.stopped:
                print('server has been stopped !')
                return False
            if (
                self.server_cmd_ops[0].output is not None 
                and 'The server is fired up and ready to roll' in ' '.join(self.server_cmd_ops[0].output)
            ):
                break
            time.sleep(1)
        return True
    
    def bench_test(self):
        if self.stopped:
            return
        for i, one_bench in enumerate(self.bench_serving):
            if self.stopped:
                return
            self.start_timer()
            self.current_bench_op = OperationContent(
                id=get_next_op_id(),
                type=OperationType.RUN,
                cmd=one_bench,
                store_output=True,
                print_output=False,
            )
            self.connection.run_cmd(self.nodes_used[0], self.current_bench_op)
            if self.stopped:
                return
            
            self.current_bench_op = None
            self.stop_timer()


class TaskOffline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        envs: Optional[Dict]
    ) -> None:
        pass
    
    def run(self):
        self.check_abnormal()
        self.start_server()
        self.stop_server()
