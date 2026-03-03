from typing import Optional, List, Dict, Any, Union
import time
import threading
import copy

from src.connection import Connection
from src.output import OutputManager, RealProgressManager
from src.benchmark import PerfBenchmark
from utils.utils import *


class BaseTask:
    def __init__(self,
        connection: Optional[Connection],
        launch_server: str,
        envs: Optional[Dict],
        task_id: int,
        dist_port: int,
        server_port: int,
    ) -> None:
        self.connection = connection
        self.timer = None
        self.launch_mode = None
        self.task_name = None
        self.task_id = task_id
        self.server_cmd = launch_server
        self.server_full_cmd = self.server_cmd
        self.server_port = server_port
        self.dist_port = dist_port
        self.output_manager: OutputManager = None
        self.real_progress_manager = None
        self.envs = envs
        self.timeout = None
        self.timer = None
        self.is_bench_finish = True

        self.all_nodes = self.connection.nodes_info
        self.nodes_used = []
        self.server_cmd_ops = []
        self.current_bench_op = None

        self.model_name = None

        self.global_logger = None
        self.logger = None

        self.stop_lock = threading.Lock()
        self._get_nodes_used()
        self.is_stopped = False
        self.log_file_subpath = None

        self.server_args_str = None
        self.server_args_dict = None
        self.failed_reason = 'Unknown Error'

        self.wake_up_event = threading.Event()

    def init(self):
        pass

    def run(self, client_id=0):
        pass

    def start_server(self):
        self.is_stopped = False
        self.wake_up_event.clear()
        self.logger = self.output_manager.logger
        nodes_num = len(self.nodes_used)
        server_cmd_port = self.server_cmd + f' --port {self.server_port} '
        for index, node in enumerate(self.nodes_used):
            cmd = server_cmd_port
            if nodes_num > 1:
                cmd += f' --dist-init-addr {self.nodes_used[0].ip}:{self.dist_port} --nnodes {nodes_num} --node-rank {index}'
            if index == 0:
                self.server_full_cmd = cmd
            output_depends = copy.deepcopy(self.output_manager.output_depends)
            output_depends.node_id = index
            output_depends.node_ip = node.ip
            self.connection.sync_output_info(node, output_depends.to_json(), self.logger)
            self.connection.check_node_gpu_in_use(node, self.logger)

            envs = self.envs.copy()
            if node.interface not in [None, '']:
                envs['GLOO_SOCKET_IFNAME'] = node.interface
            if node.ib_hcas not in [None, '']: 
                envs['MCCL_IB_HCA'] = node.ib_hcas
            op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.RUN_CMD,
                envs=envs,
                cmd=cmd,
                store_output=True,
                is_ready=True,
                is_async=True,
                is_master=index == 0,
            )
            self.logger.info(f'[{node.ip}] start run [{cmd}]')
            self.connection.run_cmd(node, op_content, self.logger, self.wake_up_event)
            self.server_cmd_ops.append(op_content)

    def stop_all(self, normal=True):
        with self.stop_lock:
            if self.is_stopped:
                return
            if not normal:
                self.stop_client()

            self.stop_server()
            if self.nodes_used[0].is_local:
                kill_local_defunct_process(self.logger)

    def stop_server(self):
        for node, op in zip(self.nodes_used, self.server_cmd_ops):
            self.connection.stop_cmd(node, op, self.logger)
        self.is_stopped = True
        self.wake_up_event.set()

    def stop_client(self):
        if self.current_bench_op:
            if self.nodes_used[0].is_local:
                self.logger.error('start kill bench client')
                kill_process_all(self.current_bench_op.handle, self.logger)
                self.logger.error('kill bench client success')
            else:
                self.logger.error(f'start kill {self.nodes_used[0].ip} bench client')
                self.connection.stop_cmd(self.nodes_used[0], self.current_bench_op, self.logger)
                self.logger.error(f'kill {self.nodes_used[0].ip} bench client success')

    def wait_server_ready(self) -> bool:
        pass

    def generate_result(self):
        pass
    
    def set_real_progress_manager(self, real_progress_manager: RealProgressManager):
        self.real_progress_manager = real_progress_manager

    def set_output_manager(self, output_manager: OutputManager) -> None:
        self.output_manager = output_manager
        self.logger = self.output_manager.logger

    def check_abnormal(self):
        abnormal_thread = threading.Thread(target=self.check_other_abnormal)
        abnormal_thread.daemon = True
        abnormal_thread.start()

    def check_other_abnormal(self):
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

        while len(self.server_cmd_ops) == 0:
            if self.is_stopped:
                return
            self.wake_up_event.wait(timeout=10)
            continue

        node_num = len(self.nodes_used)
        cur_time = time.time()
        last_log_folder_size = [0] * node_num
        curr_log_folder_size = [0] * node_num
        last_log_update_time = [cur_time] * node_num
        while True:
            abnormal_flag = False
            for index in range(0, len(self.nodes_used)):
                node = self.nodes_used[index]
                if node.is_local:
                    curr_log_folder_size[0] = get_file_size(self.output_manager.log_file_path)
                    if self.server_cmd_ops[index].output is None:
                        continue
                    output = "".join(self.server_cmd_ops[index].output)
                    for abnormal_str in abnormal_flag_str:
                        if abnormal_str in output:
                            abnormal_flag = True
                            self.logger.error(f"********************************abnormal********************************")
                            self.logger.error(f"****************************{abnormal_str}****************************")
                            self.failed_reason = abnormal_str
                            break
                    if abnormal_flag:
                        break
                else:
                    if self.is_stopped:
                        break
                    server_cmd = self.server_cmd_ops[index].cmd
                    op_content = MsgContent(
                        id=get_next_op_id(),
                        type=MsgType.CHECK_OUTPUT_FLAG,
                        cmd=server_cmd,
                        info=abnormal_flag_str
                    )
                    message = self.connection.send_slave_msg(node, op_content, self.logger)
                    output = MsgContent.from_json(message)
                    abnormal_str, curr_log_folder_size[index] = output.info
                    if abnormal_str is None:
                        continue
                    abnormal_flag = True
                    self.logger.error(f"********************************{node.ip} abnormal********************************")
                    self.logger.error(f"****************************{node.ip} {abnormal_str}****************************")
                    self.failed_reason = abnormal_str
                    break

            if abnormal_flag:
                break

            if self.is_stopped:
                self.logger.info(f"********************************server stopped, break********************************")
                break

            for node_id in range(node_num):
                if curr_log_folder_size[node_id] is not None and curr_log_folder_size[node_id] > last_log_folder_size[node_id]:
                    last_log_update_time[node_id] = time.time()
                    last_log_folder_size[node_id] = curr_log_folder_size[node_id]

            current_time = time.time()
            time_diff = [current_time - lastime for lastime in last_log_update_time]
            ip_time_diff = [f'{ip} {diff:.4f}' for ip, diff in zip(self.get_nodes_ips(), time_diff)]
            print(f"## check abnormal time_diff: {ip_time_diff}")
            self.wake_up_event.wait(timeout=10)
            if min(time_diff) >= self.timeout:
                self.logger.error(f"********************************abnormal********************************")
                self.logger.error(f"****************************timeout****************************")
                self.failed_reason = 'timeout'
                break

        self.stop_all(False)
        self.logger.info(f"check_other_abnormal exit!")

    def get_node_num_need(self):
        expected_gpu_count = self._get_gpu_count(self.server_cmd)
        node_num = -(-expected_gpu_count // self.connection.gpu_num_per_node)
        return node_num
    
    def set_node_used(self, node_pos, node_num):
        assert (
            (node_pos + node_num) <= len(self.all_nodes), 
            f'node index out of bound {node_pos=} {node_num=} total={len(self.all_nodes)}'
        )
        self.nodes_used = self.all_nodes[node_pos : (node_pos + node_num)]
        local_index = 0
        for i, node in enumerate(self.nodes_used):
            if node.is_local:
                local_index = i
                break
        if local_index > 0:
            self.nodes_used[0], self.nodes_used[local_index] = self.nodes_used[local_index], self.nodes_used[0]

    def get_nodes_ips(self):
        return [node.ip for node in self.nodes_used]

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
            'pp-size': None
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
        pp = params["pp-size"]
        gpu_count = 0
        if pp:
            gpu_count = pp*tp
        else:
            gpu_count = tp
        return gpu_count
    
    def convert_server_args_to_dict(self, args_str):
        args_dict = {}
        for param in args_str.split(','):
            param_match = re.match(r'(\w+)=(.*)', param.strip())
            if param_match:
                key = param_match.group(1)
                value = param_match.group(2)
                args_dict[key] = value
        return args_dict


class TaskOnline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        bench_serving,
        task_id: int,
        envs: Optional[Dict],
        dist_port: int,
        server_port: int,
    ) -> None:
        super().__init__(connection, launch_server, envs, task_id, dist_port, server_port)
        self.bench_serving = bench_serving
        self.launch_mode = TaskLaunchMode.online
        self.current_bench_id = 0
        self.bench_total_num = sum([len(bench.cmd_list) for bench in self.bench_serving])
        self.server_info = {}

    def init(self):
        self.is_stopped = False
        self.server_cmd_ops.clear()
        self.wake_up_event.clear()

    def run(self, client_id=0):
        while True:
            self.init()
            self.check_abnormal()
            self.start_server()
            self.real_progress_manager.init_task_content(self)
            self.is_bench_finish = False
            server_start = self.wait_server_ready()
            if server_start and not self.is_bench_finish:
                if self.nodes_used[0].is_local:
                    assert(self.server_cmd_ops[0].output is not None, f'Task {self.task_name}_server{self.task_id} output is none')
                    self.server_args_str, args_str = match_server_args(' '.join(self.server_cmd_ops[0].output), self.server_cmd_ops[0].cmd, self.logger)
                    self.server_args_dict = self.convert_server_args_to_dict(args_str)
                else:
                    op_content = MsgContent(
                        id=get_next_op_id(),
                        type=MsgType.GET_SERVER_ARGS,
                        cmd=self.server_cmd_ops[0].cmd,
                    )
                    message = self.connection.send_slave_msg(self.nodes_used[0], op_content, self.logger)
                    output = MsgContent.from_json(message)
                    self.server_args_str, args_str = output.info[0], output.info[1]
                    self.server_args_dict = self.convert_server_args_to_dict(args_str)
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
        ready_flag = 'The server is fired up and ready to roll'
        if self.nodes_used[0].is_local:
            while True:
                if self.is_stopped:
                    self.logger.error('server has been stopped !')
                    return False
                ready = False
                if self.nodes_used[0].is_local:
                    ready = (
                        self.server_cmd_ops[0].output is not None 
                        and ready_flag in ''.join(self.server_cmd_ops[0].output)
                    )
                if ready:
                    pattern = r'available_gpu_mem=(\d+\.\d+)\s*GB'
                    for line in reversed(self.server_cmd_ops[0].output):
                        match = re.search(pattern, line)
                        if match:
                            self.output_manager.server_available_gpu_mem = match.group(1) + " GB"
                            break
                    break
                self.wake_up_event.wait(timeout=10)
            return True
        else:
            op_content = MsgContent(
                id=get_next_op_id(),
                type=MsgType.GET_AVAILABLE_MEM,
                cmd=self.server_cmd_ops[0].cmd,
                info=[ready_flag]
            )
            message = self.connection.send_slave_msg(self.nodes_used[0], op_content, self.logger)
            if len(message) == 0:
                return False
            self.output_manager.server_available_gpu_mem = message


    def bench_test(self, client_id=0):
        # if self.is_stopped:
        #     return

        bench_id = -1
        for benchmark in self.bench_serving:
            for bench_cmd in benchmark.cmd_list:
                bench_id += 1
                # if self.is_stopped:
                #     self.logger.error(f'bench stop success0')
                #     return

                # 此处判断如果上次bench中断,则跳过之前已经跑过的bench client
                if bench_id < self.current_bench_id:
                    continue
                self.current_bench_id = bench_id + 1

                # 此处判断如果上次主进程中断,则跳过之前已经跑过的bench client
                if bench_id < client_id:
                    self.logger.info("current client id " + str(bench_id) + " last client id " + str(client_id))
                    continue
                
                # --output-file 追加, 不然jsonl生成到其他目录下
                one_bench_with_output = self.output_manager.append_result_file_param(bench_cmd, benchmark)

                self.current_bench_op = MsgContent(
                    id=get_next_op_id(),
                    type=MsgType.RUN_CMD,
                    envs=benchmark.envs,
                    cmd=one_bench_with_output,
                    store_output=True,
                    print_output=True,
                    is_master=True,
                    is_benching=True,
                    special_logger=self.global_logger if self.nodes_used[0].is_local else None
                )

                self.current_bench_id = bench_id + 1
                self.real_progress_manager.write_real_progress_bench_serving(self.task_id, bench_id, bench_cmd)
                if not self.is_stopped:
                    self.connection.run_cmd(self.nodes_used[0], self.current_bench_op, self.logger, self.wake_up_event)
                status = self.current_bench_op.status
                result = self.current_bench_op.output

                if status == 0:
                    self.real_progress_manager.write_real_progress_result('pass', self, bench_id)
                    self.output_manager.write_client_result(bench_cmd, benchmark, result)
                else:
                    self.real_progress_manager.write_real_progress_result('fail', self, bench_id)
                    self.output_manager.write_client_result(bench_cmd, benchmark, result, False)

                self.output_manager.extract_result_metrics(bench_cmd, benchmark, self.server_args_str, self.server_args_dict)

                # if self.is_stopped:
                #     self.logger.error(f'bench stop success1')
                #     if self.current_bench_id == len(self.bench_serving):
                #         self.is_bench_finish = True
                #     return

                self.current_bench_op = None

        if self.current_bench_id == self.bench_total_num:
            self.is_bench_finish = True


class TaskOffline(BaseTask):
    def __init__(self,
        connection: Connection,
        launch_server: str,
        task_id: int,
        envs: Optional[Dict],
        dist_port: int,
        server_port: int,
    ) -> None:
        super().__init__(connection, launch_server, envs, task_id, dist_port, server_port)
        self.launch_mode = TaskLaunchMode.offline
        self.task_id = task_id
        self.bench_serving = PerfBenchmark('custom', None, task_id, TaskLaunchMode.offline)
        self.bench_total_num = 1
    
    def start_offline_server(self):
        # 拼接 --result-filename = xxxxx.jsonl
        self.server_cmd  = self.output_manager.append_result_file_param(self.server_cmd, self.bench_serving)
        self.start_server()
        self.real_progress_manager.init_task_content(self)
        if self.nodes_used[0].is_local:
            master_ops = self.server_cmd_ops[0]
            master_ops.thread.join()
            status = master_ops.status
            result = master_ops.output
        else:
            while True:
                get_op = MsgContent(
                    id=get_next_op_id(),
                    type=MsgType.GET_CMD_STATUS,
                    cmd=self.server_cmd_ops[0].cmd,
                )
                message = self.connection.send_slave_msg(self.nodes_used[0], get_op, self.logger)
                output_msg = MsgContent.from_json(message)
                if output_msg.status is not None:
                    status = output_msg.status
                    result = output_msg.output
                    break
                is_set = self.wake_up_event.wait(timeout=10)
                if is_set:
                    break
        
        if status == 0:
            self.real_progress_manager.write_real_progress_result('pass',self)
            self.output_manager.write_client_result(self.server_cmd, self.bench_serving, result)

            self.server_args_str, args_str = match_server_args(' '.join(result), self.server_cmd_ops[0].cmd, self.logger)
            self.server_args_dict = self.convert_server_args_to_dict(args_str)
            self.output_manager.extract_result_metrics(self.server_cmd, self.bench_serving, self.server_args_str, self.server_args_dict)
        else:
            self.real_progress_manager.write_real_progress_result('fail',self)
            self.output_manager.write_client_result(self.server_cmd, self.bench_serving, result, False)
        
    def run(self, client_id=0):
        self.check_abnormal()
        self.start_offline_server()
        self.stop_server()
