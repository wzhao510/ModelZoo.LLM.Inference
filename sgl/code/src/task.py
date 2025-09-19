from typing import Optional, List, Dict, Any, Union
import time
import threading

from src.connection import Connection
from src.output import BenchmarkOutputManager, AccOutputManager, SearchOutputManager, RampupOutputManager, RealProgressManager, NOW_TIME
from utils.utils import *


TIMEOUT_DURATION = 60*10.678

class BaseTask:
    def __init__(self,
        connection: Optional[Connection],
        launch_server: str,
        envs: Optional[Dict],
        task_id: int,
        port: int = 5000
    ) -> None:
        self.connection = connection
        self.timer = None
        self.launch_mode = None
        self.task_name = None
        self.task_type = None
        self.task_id = task_id
        self.server_cmd = launch_server
        self.server_full_cmd = self.server_cmd
        self.server_port = port
        self.output_manager: Union["BenchmarkOutputManager","AccOutputManager", "SearchOutputManager", "RampupOutputManager"] = None
        self.real_progress_manager = None
        self.envs = envs
        self.timeout = None
        self.timer = None
        self.is_bench_finish = True
        self.is_kill_abnormal = False

        # todo not use all node
        self.all_nodes = self.connection.nodes_info
        self.nodes_used = []
        self.server_cmd_ops = []
        self.current_bench_op = None

        self.model_name = None
        self.acc_type = None
        self.max_ttft = None
        self.max_tpot = None

        self.global_logger = None
        self.logger = None

        self.stop_lock = threading.Lock()
        self._get_nodes_used()
        self.is_stopped = False
        self.log_file_subpath = None

    def init(self):
        pass

    def run(self, client_id=0):
        pass

    def start_server(self):
        self.is_stopped = False
        self.logger = self.output_manager.logger
        self.connection.task_logger = self.logger
        nodes_num = len(self.nodes_used)
        if nodes_num == 1:
            printenv(self.logger)
            check_gpu_in_use(self.logger)
            op_content = OperationContent(
                        id=get_next_op_id(),
                        type=OperationType.RUN,
                        envs=self.envs,
                        cmd=self.server_cmd,
                        store_output=True,
                        is_ready=True,
                        is_async=True,
                        is_master=True,
                    )
            self.connection.run_cmd(self.nodes_used[0], op_content)
            self.server_cmd_ops.append(op_content)
        else:            
            # 先做一遍全节点检查，目前只是集中了日志，还没有实际行为
            for index, node in enumerate(self.nodes_used):
                if index == 0:
                    printenv(self.logger)
                    check_gpu_in_use(self.logger)
                else:
                    cmd = self.server_cmd + f' --nnodes {nodes_num} --node-rank {index}'
                    self.connection.init_slave_output_file(node, self.output_manager.get_info_for_slave(), cmd)
                    self.connection.check_slave_gpu_in_use(node)

            for index, node in enumerate(self.nodes_used):
                cmd = self.server_cmd
                cmd += f' --dist-init-addr {self.nodes_used[0].ip}:{self.server_port} --nnodes {nodes_num} --node-rank {index}'
                if index == 0:
                    self.server_full_cmd = cmd

                op_content = OperationContent(
                    id=get_next_op_id(),
                    type=OperationType.RUN,
                    envs=self.envs,
                    cmd=cmd,
                    store_output=True,
                    is_ready=True,
                    is_async=True,
                    is_master=index == 0,
                )
                self.connection.run_cmd(node, op_content)
                self.server_cmd_ops.append(op_content)

    def stop_all(self, normal=True):
        with self.stop_lock:
            if self.is_stopped:
                return
            if not normal:
                self.stop_client()

            self.stop_server()
            kill_local_defunct_process(self.logger)

    def stop_server(self):
        for node, op in zip(self.nodes_used, self.server_cmd_ops):
            self.connection.stop_cmd(node, op)
        self.is_stopped = True

    def stop_client(self):
        if self.current_bench_op:
            self.logger.error('start kill bench client')
            kill_process_all(self.current_bench_op.handle, self.logger)
            self.logger.error('kill bench client success')

    def wait_server_ready(self) -> bool:
        pass

    def generate_result(self):
        pass
    
    def set_real_progress_manager(self, real_progress_manager: RealProgressManager):
        self.real_progress_manager = real_progress_manager

    def set_output_manager(self, output_manager: Union["BenchmarkOutputManager","AccOutputManager", "SearchOutputManager", "RampupOutputManager"]) -> None:
        self.output_manager = output_manager
        self.logger = self.output_manager.logger

    def set_is_kill_abnormal(self, is_kill_abnormal):
        self.is_kill_abnormal = is_kill_abnormal

    def check_abnormal(self):
        abnormal_thread = threading.Thread(target=self.check_other_abnormal)
        abnormal_thread.daemon = True
        abnormal_thread.start()

    def set_file_log_subfile(self, subpath):
        self.log_file_subpath = subpath

    def check_other_abnormal(self):
        last_log_folrder_size = 0
        abnormal_flag_str = [
            "Gracefully exiting... remaining number of requests",
            "Watchdog timeout (self.watchdog_timeout=300)",
            "torch.OutOfMemoryError: CUDA out of memory.",
            "Forcing disable 'CUTLASS' backend as it is not supported in maca platform.",
            "TypeError: launcher() got an unexpected keyword argument 'scenario'",
            "Exception: Capture cuda graph failed:",
            "RuntimeError: CUDA error: out of memory",
            "RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.",
            "ModuleNotFoundError: No module named 'flashinfer'",
            "CUDA error: an illegal memory access was encountered",
            "RuntimeError: NCCL error: internal error",
            "CUDA error: invalid device ordinal",
            "ImportError: cannot import name 'layer_type_validation'",
            "RuntimeError: The server socket has failed to listen on any local network address",
            "huggingface_hub.errors.HFValidationError: Repo id must be in the form",
            "OSError: Can't load the configuration of",
        ]

        last_log_update_time = time.time()
        while len(self.server_cmd_ops) == 0:
            time.sleep(10)
            continue

        while True:
            abnormal_flag = False
            output = "".join(self.server_cmd_ops[0].output)
            for abnormal_str in abnormal_flag_str:
                if abnormal_str in output:
                    abnormal_flag = True
                    self.logger.error(f"********************************abnormal********************************")
                    self.logger.error(f"****************************{abnormal_str}****************************")
                    self.real_progress_manager.set_fail_reason(abnormal_str)
                    break
            if abnormal_flag:
                break

            if self.is_stopped:
                self.logger.info(f"********************************server stopped, break********************************")
                break

            for index in range(1, min(len(self.nodes_used), len(self.server_cmd_ops))):
                node = self.nodes_used[index]
                server_cmd = self.server_cmd_ops[index].cmd
                op_content = OperationContent(
                    id=get_next_op_id(),
                    type=OperationType.CHECK_ABNORMAL,
                    cmd=server_cmd,
                    abnormal_flags=abnormal_flag_str
                )
                message = self.connection.run_cmd(node, op_content)
                output = OperationContent.from_json(message)
                if output.abnormal_flags is None:
                    continue
                abnormal_str = output.abnormal_flags[0]
                abnormal_flag = True
                self.logger.error(f"********************************{node.ip} abnormal********************************")
                self.logger.error(f"****************************{node.ip} {abnormal_str}****************************")
                self.real_progress_manager.set_fail_reason(abnormal_str)
                break

            if abnormal_flag:
                break

            file_size = self.get_folder_size(self.log_file_subpath) 
            if file_size > last_log_folrder_size:
                last_log_update_time = time.time()
                last_log_folrder_size = file_size

            current_time = time.time()
            time_diff = current_time - last_log_update_time
            print(f"## check abnormal time_diff: {time_diff}")
            time.sleep(10)
            if time_diff >= TIMEOUT_DURATION:
                self.logger.error(f"********************************abnormal********************************")
                self.logger.error(f"****************************timeout****************************")
                self.real_progress_manager.set_fail_reason('timeout')
                break

        self.stop_all(False)
        self.logger.info(f"check_other_abnormal exit!")

    def get_folder_size(self, folder_path):
        total_size = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    def set_server_port(self, port):
        self.server_port = port

    def _get_nodes_used(self):
        expected_gpu_count = self._get_gpu_count(self.server_cmd)
        node_num = -(-expected_gpu_count // self.connection.gpu_num_per_node)
        self.nodes_used = [None] * min(node_num, len(self.all_nodes))
        slave_node = 1
        for node in self.all_nodes:
            if node.is_local:
                self.nodes_used[0] = node
                continue
            if slave_node < node_num:
                self.nodes_used[slave_node] = node
                slave_node += 1


    def _get_gpu_count(self,command_str):
        # 初始化要提取的参数
        params = {
            'tp': None,
            'dp': None,
            'pp': None
        }
        
        # 分割命令字符串为参数列表
        parts = command_str.split()
        
        # 遍历参数列表查找目标参数
        for i in range(len(parts)):
            for param in params.keys():
                if parts[i] == f'--{param}':
                    # 确保有下一个元素作为值
                    if i + 1 < len(parts):
                        params[param] = int(parts[i + 1])
        
        tp = params["tp"]
        pp = params["pp"]
        gpu_count = 0
        if pp:
            gpu_count = pp*tp
        else:
            gpu_count = tp
        return gpu_count

class TaskOnline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        bench_serving: List[str],
        task_id: int,
        envs: Optional[Dict],
        port: int
    ) -> None:
        super().__init__(connection, launch_server, envs, task_id)
        super().set_server_port(port)
        self.bench_serving = bench_serving
        self.launch_mode = TaskLaunchMode.online
        self.current_bench_id = 0

    def init(self):
        self.is_stopped = False
        self.server_cmd_ops.clear()

    def run(self, client_id=0):
        while True:
            self.init()
            self.check_abnormal()
            self.start_server()
            self.real_progress_manager.init(self)
            self.real_progress_manager.write_real_progress_bench_serving(0, False)
            self.is_bench_finish = False
            server_start = self.wait_server_ready()
            if server_start and not self.is_bench_finish:
                self.bench_test(client_id)

            # 判断运行过程中是否出现错误,导致测试中断
            # 1.未出现错误中断,退出
            # 2.出现错误中断,重新启动server,从中断处继续测试
            self.stop_all()
            time.sleep(10)
            self.logger.info(f'stop server success')
            if not server_start or self.is_bench_finish:
                break

    def wait_server_ready(self):
        while True:
            if self.is_stopped:
                self.logger.error('server has been stopped !')
                return False
            if (
                self.server_cmd_ops[0].output is not None 
                and 'The server is fired up and ready to roll' in ' '.join(self.server_cmd_ops[0].output)
            ):
                break
            time.sleep(1)
        return True

    def bench_test(self, client_id=0):
        if self.is_stopped:
            return

        #for i, one_bench in enumerate(self.bench_serving):
        bench_id = -1
        for content in self.bench_serving:
            bench_id += 1
            if self.is_stopped:
                self.logger.error(f'bench stop success0')
                return

            # 此处判断如果上次bench中断,则跳过之前已经跑过的bench client
            if bench_id < self.current_bench_id:
                continue
            self.current_bench_id = bench_id + 1

            # 此处判断如果上次主进程中断,则跳过之前已经跑过的bench client
            if bench_id < client_id:
                self.logger.info("current client id " + str(bench_id) + " last client id " + str(client_id))
                continue
            
            # --output-file 追加, 不然jsonl生成到其他目录下
            one_bench_with_output = self.output_manager.append_result_file_param(content.get_cmd())

            self.current_bench_op = OperationContent(
                id=get_next_op_id(),
                type=OperationType.RUN,
                envs=self.envs,
                cmd=one_bench_with_output,
                store_output=True,
                print_output=True,
                is_master=True,
                is_benching=True,
                special_logger=self.global_logger
            )

            self.current_bench_id = bench_id + 1
            self.real_progress_manager.write_real_progress_bench_serving(bench_id)
            self.connection.run_cmd(self.nodes_used[0], self.current_bench_op)
            statu = self.current_bench_op.status
            result = self.current_bench_op.output

            if statu == 0:
                self.real_progress_manager.write_real_progress_result('pass', content.get_id())
                self.output_manager.write_client_result(content.get_cmd(), result)
            else:
                self.real_progress_manager.write_real_progress_result('fail', content.get_id())
                self.output_manager.write_client_result(content.get_cmd(), result, False)

            if self.is_stopped:
                self.logger.error(f'bench stop success1')
                if self.current_bench_id == len(self.bench_serving):
                    self.is_bench_finish = True
                return

            self.current_bench_op = None

        self.output_manager.extract_result_metrics()
        if self.task_type == TaskType.search:
            self.output_manager.parser_single_search_data()
        if self.current_bench_id == len(self.bench_serving):
            self.is_bench_finish = True
            self.real_progress_manager.write_to_run_args(self.task_id+1, 0)

class TaskOffline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        task_id: int,
        envs: Optional[Dict],
        port: int
    ) -> None:
        super().__init__(connection, launch_server, envs, task_id)
        super().set_server_port(port)        
        self.launch_mode = TaskLaunchMode.offline
        self.task_id = task_id
    
    def start_offline_server(self):
        # 拼接 --result-filename = xxxxx.jsonl
        self.server_cmd  = self.output_manager.append_result_file_param(self.server_cmd)
        self.start_server()
        self.real_progress_manager.init(self)
        master_ops = [ops for ops in self.server_cmd_ops if ops.is_master][0]
        master_ops.thread.join()
        statu = master_ops.status
        result = master_ops.output
       
        if statu == 0:
            self.real_progress_manager.write_real_progress_result('pass',self.task_id)
            self.output_manager.write_client_result(self.server_cmd, result)
        else:
            self.real_progress_manager.write_real_progress_result('fail',self.task_id)
            self.output_manager.write_client_result(self.server_cmd, result, False)
        self.output_manager.extract_result_metrics()
        
        
    def run(self, client_id=0):
        self.check_abnormal()
        self.start_offline_server()
        self.stop_server()
