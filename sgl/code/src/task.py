from typing import Optional, List, Dict, Any
import time
import threading

from src.connection import Connection
from src.output import OutputManager
from utils.utils import *


TIMEOUT_DURATION = 60*10.678
G_MASTER_LOCK = threading.Lock()
last_log_folrder_size = 0
is_abnormal_start = False
test_stop = False
abnormal_flag = False

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
        self.server_port = port
        self.output_manager = None
        self.envs = envs
        self.timeout = None
        self.timer = None
        self.is_bench_finish = True
        self.is_kill_abnormal = False

        # todo not use all node
        self.nodes_used = self.connection.nodes_info
        self.server_cmd_ops = []
        self.current_bench_op = None

        self.model_name = None
        self.acc_type = None
        self.max_ttft = None
        self.max_tpot = None

    def run(self, client_id=0):
        pass

    def start_server(self):
        nodes_num = len(self.nodes_used)
        
        # 先做一遍全节点检查，目前只是集中了日志，还没有实际行为 by ydm.
        for index, node in enumerate(self.nodes_used):
            if index == 0:
                printenv(self.envs)
                check_gpu_in_use(self.envs)
            else:
                cmd = self.server_cmd + f'  --nnodes {nodes_num} --node-rank {index}'
                self.connection.init_slave_output_file(node, self.output_manager.get_info_for_slave(), cmd)
                self.connection.check_slave_gpu_in_use(node)

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
                is_master=index == 0,
            )
            self.connection.run_cmd(node, op_content)
            self.server_cmd_ops.append(op_content)

    def stop_all(self, normal=True):
        if not normal:
            self.stop_client()

        self.stop_server()
        op_content = OperationContent(
                id=get_next_op_id(),
                type=OperationType.RUN,
                cmd="pkill -9 -f sglang",
                store_output=True,
                is_ready=True,
                is_async=True,
                is_master=True,
                info=self.output_manager.get_info_for_slave(),
        )
        run_sys_cmd(op_content)

    def stop_server(self):
        global abnormal_flag
        if self.is_kill_abnormal:
            abnormal_flag = True
        for node, op in zip(self.nodes_used, self.server_cmd_ops):
            self.connection.stop_cmd(node, op)
        global test_stop
        test_stop = True

    def stop_client(self):
        if self.current_bench_op:
            logger.error('start kill bench client')
            kill_process_all(self.current_bench_op.handle)
            logger.error('kill bench client success')

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
    
    def set_output_manager(self, output_manager: OutputManager,is_kill_abnormal) -> None:
        self.output_manager = output_manager
        self.is_kill_abnormal = is_kill_abnormal

    def check_abnormal(self):
        global last_log_folrder_size, is_abnormal_start
        last_log_folrder_size = 0
        if not is_abnormal_start:
            abnormal_thread = threading.Thread(target=self.check_other_abnormal)
            abnormal_thread.daemon = True
            abnormal_thread.start()
            is_abnormal_start = True

    def check_other_abnormal(self):
        global last_log_folrder_size,is_abnormal_start,abnormal_flag
        abnormal_flag_str = ["Gracefully exiting... remaining number of requests",
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
                             "CUDA error: invalid device ordinal"
                            ]

        last_log_update_time = time.time()
        abnormal_flag = False
        while len(self.server_cmd_ops) == 0:
            time.sleep(10)
            continue

        self.server_cmd_ops[0].output.clear()
        while not abnormal_flag:
            for abnormal_str in abnormal_flag_str:
                output = " ".join([s for s in self.server_cmd_ops[0].output])
                if abnormal_str in output:
                    abnormal_flag = True
                    logger.error(f"********************************abnormal********************************")
                    logger.error(f"****************************{abnormal_str}****************************")
                    self.output_manager.set_fail_reason(abnormal_str)
                    break
            if abnormal_flag:
                break

            file_size = self.get_folder_size(self.output_manager.get_log_path()) 
            if file_size > last_log_folrder_size:
                last_log_update_time = time.time()
                last_log_folrder_size = file_size

            current_time = time.time()
            time_diff = current_time - last_log_update_time
            print(f"## check abnormal time_diff: {time_diff}")
            time.sleep(10)
            if time_diff >= TIMEOUT_DURATION:
                abnormal_flag = True
                logger.error(f"********************************abnormal********************************")
                logger.error(f"****************************timeout****************************")
                self.output_manager.set_fail_reason('timeout')

        is_abnormal_start = False
        self.stop_all(False)

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
        global test_stop
        test_stop = False
        self.server_cmd_ops.clear()

    def run(self, client_id=0):
        while True:
            self.init()
            self.check_abnormal()
            self.output_manager.write_real_progress_bench_serving(0, False)
            self.start_server()
            self.is_bench_finish = False
            server_start = self.wait_server_ready()
            if server_start and not self.is_bench_finish:
                self.bench_test(client_id)

            # 判断运行过程中是否出现错误,导致测试中断
            # 1.未出现错误中断,退出
            # 2.出现错误中断,重新启动server,从中断处继续测试
            self.stop_all()
            time.sleep(10)
            logger.info(f'stop server success')
            if not server_start or self.is_bench_finish:
                break

    def wait_server_ready(self):
        global test_stop
        while True:
            if test_stop:
                logger.error('server has been stopped !')
                return False
            if (
                self.server_cmd_ops[0].output is not None 
                and 'The server is fired up and ready to roll' in ' '.join(self.server_cmd_ops[0].output)
            ):
                break
            time.sleep(1)
        return True

    def bench_test(self, client_id=0):
        global test_stop
        if test_stop:
            return

        #for i, one_bench in enumerate(self.bench_serving):
        bench_id = -1
        for content in self.bench_serving:
            bench_id += 1
            if test_stop:
                logger.error(f'bech stop success0')
                return

            # 此处判断如果上次bench中断,则跳过之前已经跑过的bench client
            if bench_id < self.current_bench_id:
                continue
            self.current_bench_id = bench_id + 1

            # 此处判断如果上次主进程中断,则跳过之前已经跑过的bench client
            if bench_id < client_id:
                logger.info("current client id " + str(bench_id) + " last client id " + str(client_id))
                continue

            # --output-file 追加, 不然jsonl生成到其他目录下
            one_bench_with_output = self.output_manager.append_result_file_param(content.get_cmd())

            self.current_bench_op = OperationContent(
                id=get_next_op_id(),
                type=OperationType.RUN,
                cmd=one_bench_with_output,
                store_output=True,
                print_output=True,
                is_master=True,
                is_benching=True
            )

            self.current_bench_id = bench_id + 1
            self.output_manager.write_real_progress_bench_serving(bench_id)
            self.connection.run_cmd(self.nodes_used[0], self.current_bench_op)
            statu = self.current_bench_op.status
            result = self.current_bench_op.output

            if statu == 0:
                self.output_manager.write_real_progress_result('pass', content.get_id())
                self.output_manager.write_client_result(content.get_cmd(), result)
            else:
                self.output_manager.write_real_progress_result('fail', content.get_id())
                self.output_manager.write_client_result(content.get_cmd(), result, False)

            if test_stop:
                logger.error(f'bech stop success1')
                return

            self.current_bench_op = None

        self.output_manager.extract_result_metrics()
        if self.task_type == TaskType.search:
            self.output_manager.parser_single_search_data()
        if self.current_bench_id == len(self.bench_serving):
            self.is_bench_finish = True
            self.output_manager.write_to_run_args(self.task_id+1, 0)

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
        master_ops = [ops for ops in self.server_cmd_ops if ops.is_master][0]
        master_ops.thread.join()
        statu = master_ops.status
        result = master_ops.output
       
        if statu == 0:
            self.output_manager.write_real_progress_result('pass',self.task_id)
            self.output_manager.write_client_result(self.server_cmd, result)
        else:
            self.output_manager.write_real_progress_result('fail',self.task_id)
            self.output_manager.write_client_result(self.server_cmd, result, False)
        self.output_manager.extract_result_metrics()
        
        
    def run(self, client_id=0):
        self.check_abnormal()
        self.start_offline_server()
        self.stop_server()
