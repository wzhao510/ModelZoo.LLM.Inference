import argparse
import sys
import signal
import json
import re
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from src.task import TaskOnline, TaskOffline,BaseTask

from src.connection import Connection
from src.output import OutputManager, RealProgressManager
from src.benchmark import Benchmark

from utils.utils import *
from itertools import product
from collections import defaultdict
import pandas as pd

MODEL_SERIES=['Qwen', 'DeepSeek', 'Kimi', 'GLM', 'MiniMax']
DEFAULT_SERVER_PORT = 30000
DEFAULT_DIST_PORT = 5000

class ServerParser:
    @staticmethod
    def online_server(server, logger = None):
        param_list = []
        for one_param in server:
            if isinstance(one_param, list):
                param_list.append(one_param)
            else:
                param_list.append([one_param])

        launch_server_commands = []
        for combo in list(product(*param_list)):
            full_command = ""
            for param in combo:
                full_command += f" {param.strip()}" if full_command != "" else f"{param.strip()}"
            full_command = full_command.strip()
            log_msg_level(f"\t\t|___ {full_command}", logger)
            launch_server_commands.append(full_command)
        return launch_server_commands

    @staticmethod
    def get_server_list(config, task_name, task, logger):
        server_list = []
        server_cmd = task['launch_server']
        if isinstance(server_cmd, list):
            server_list = ServerParser.online_server(server_cmd, logger=logger)
        elif isinstance(server_cmd, str):
            server_cmds = get_json_config_default(config, 'server_cmds', None)
            for cmd_name in server_cmd.split(';'):
                if server_cmds is None or cmd_name.strip() not in server_cmds.keys():
                    raise RuntimeError(f"Task[{task_name}] launch server cmd{cmd_name.strip()} not set!")
                log_msg_level(f"|—— {task_name}", logger)
                server_list.extend(ServerParser.online_server(server_cmds[cmd_name.strip()], logger=logger))
        else:
            raise RuntimeError(f"Task[{task_name}] launch server cmd only support str or list!")
        return server_list

    @staticmethod
    def extract_task_model_name(server_cmd) -> str:
        unknown_model_name = 'UNKNOWN'
        model_path = re.search(r"--model-path\s+(\S+)", server_cmd)
        if model_path is None:
            return unknown_model_name
        path = Path(model_path.group(1))
        parts = path.parts
        model_str = MODEL_SERIES
        for part in enumerate(parts[::-1]):
            for model in MODEL_SERIES:
                if model.lower() in part[1].lower():
                    return part[1]
        return parts[-1]

    @staticmethod
    def merge_task_envs(environments, task_envs, logger):
        default_envs = get_json_config_default(environments, 'default_envs', [])
        if len(task_envs) == 0:
            env = convert_str_to_env_dict(default_envs)
            log_msg_level(f"\t\t|___ {env}", logger)
            return env
        new_env = []
        for env_value in task_envs:
            if env_value in environments.keys():
                new_env.extend(environments[env_value])
                continue
            new_env.append(env_value)
        new_env = convert_str_to_env_dict(new_env)
        log_msg_level(f"\t\t|___ {new_env}", logger)
        return new_env
    

class TaskScheduler:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args:argparse.Namespace = args
        self.task_list = []
        self.current_task_ids = []
        self.finish_flag = False
        self.connection:Optional[Connection] = None
        self.is_stopped = False
        self.real_progress_manager = RealProgressManager(self.args)
        self.now_path = self.real_progress_manager.now_time_path
        self.global_logger = get_logger(self.now_path, 'bench_record.log')
        self.server_id_global = -1
        self.server_pass_list = defaultdict(list)
        self.lock = threading.Lock()

    def pass_id_filter(self):
        # 1.获取 total_real_progress_file 文件数据
        # 2.解析出服务正常和bench PASS的任务
        # 3.放入全局变量 server_pass_list, 以备后续过滤使用
        if not os.path.exists(self.real_progress_manager.total_real_progress_path):
            self.global_logger.info(f'current total file is not exists..return')
            return

        total_data = {}
        with open(self.real_progress_manager.total_real_progress_path, "r") as f:
            total_data = json.load(f)

        task_list = total_data['tasks']
        if len(task_list) == 0:
            self.global_logger.info(f'The current task list is not empty, follow the normal production task process')
            return

        for task in task_list:
            if task["launch_mode"] == "online":
                for client in task["client_test"]:
                    if client["status"] != "pass":
                        continue
                    self.server_pass_list[task['server_id']].append(client["id"])
            elif task["launch_mode"] == "offline":
                if task["status"] != "pass":
                    continue
                self.server_pass_list[task['server_id']] = []
            else:
                self.global_logger.error(f'launch_mode is unknown type...' + task["launch_mode"] + " server id: " + task["server_id"])
        self.global_logger.info("current server_pass_list after filter:")
        self.global_logger.info(self.server_pass_list)
    
    def is_valid_task_config(self, config_path, replacements):
        """
        检查是否为有效的task配置文件
        条件 1. 是有效的JSON文件 2. 包含'tasks'字段
        """
        if not config_path.lower().endswith(('.json')):
            return False, 0
        try:
            config = read_json(config_path, replacements)
            # 检查是否包含必要的tasks字段
            if 'tasks' in config and isinstance(config['tasks'], dict):
                return True,config
            else:
                self.global_logger.warning(f"Skip invalid task config: {config_path} - Missing or invalid 'tasks' field")
                return False,0
        except Exception as e:
            self.global_logger.warning(f"Skip invalid JSON file: {config_path} - Error: {str(e)}")
            return False,0

    def parse_task(self, config, task_name, task_config, environments) -> None:
        task_list = []
        server_list = ServerParser.get_server_list(config, task_name, task_config, self.global_logger)
        launch_mode = get_json_config_default(task_config, 'launch_mode', 'online')
        envs = ServerParser.merge_task_envs(environments, get_json_config_default(task_config, 'environment', []), self.global_logger)
        dist_port = get_json_config_default(task_config, 'dist_port', DEFAULT_DIST_PORT)
        server_port = get_json_config_default(task_config, 'server_port', DEFAULT_SERVER_PORT)

        client_id = 0
        benchmark_cmd = task_config['benchmark']
        benchmark_list = []
        if isinstance(benchmark_cmd, dict):
            bench_type = get_json_config_default(benchmark_cmd, 'type', 'perf')
            if bench_type in self.args.specify_test:
                benchmark_list = [Benchmark.parse_config('', benchmark_cmd, client_id, environments, launch_mode)]
        elif isinstance(benchmark_cmd, str):
            benchmark_cmds = get_json_config_default(config, 'benchmark_cmds', None)
            for bench_name in benchmark_cmd.split(';'):
                if benchmark_cmds is None or bench_name.strip() not in benchmark_cmds.keys():
                    raise RuntimeError(f"Task[{task_name}] benchmark cmd {bench_name.strip()} not set!")
                bench_type = get_json_config_default(benchmark_cmds[bench_name.strip()], 'type', 'perf')
                if bench_type in self.args.specify_test:
                    benchmark_list.append(
                        Benchmark.parse_config(bench_name.strip(), benchmark_cmds[bench_name.strip()], client_id, environments, launch_mode)
                    )
                    client_id = benchmark_list[-1].get_end_id()
        else:
            raise RuntimeError(f"Task[{task_name} launch server cmd only support str or dict!")

        for server_cmd in server_list:
            if launch_mode == 'online':
                self.server_id_global += 1
                # if str(self.server_id_global) in self.server_pass_list and len(self.server_pass_list[str(self.server_id_global)]) == client_id:
                #     continue

                # benchmark_list_filter = ConfigParser.filter_benchmark(server_id_global, benchmark_list, incremental_mode)
                _task = TaskOnline(self.connection, server_cmd, benchmark_list, self.server_id_global, envs, dist_port, server_port)
                _task.task_name = task_name
                _task.timeout = self.args.timeout
                _task.model_name = ServerParser.extract_task_model_name(server_cmd)
                _task.set_output_manager(OutputManager.from_task(self.args, _task, self.args.local_ip))
                _task.global_logger = self.global_logger
                task_list.append(_task)
            else:
                for benchmark in benchmark_list:
                    for cmd in benchmark.cmd_list:
                        self.server_id_global += 1
                        # if str(server_id_global) in server_pass_list:
                        #     continue
                        _task = TaskOffline(self.connection, f'{server_cmd} {cmd}', self.server_id_global, envs, dist_port, server_port)
                        _task.task_name = task_name
                        _task.timeout = self.args.timeout
                        _task.model_name = ServerParser.extract_task_model_name(server_cmd)
                        _task.set_output_manager(OutputManager.from_task(self.args, _task, self.args.local_ip))
                        _task.global_logger = self.global_logger
                        task_list.append(_task)
        return task_list

    def generate_task(self) -> None:
        # incremental_mode = False
        machines = read_json(self.args.machine_config)
        self.connection = Connection(machines['machine_info'], self.args.port, self.args.local_ip, logger=self.global_logger)
        self.connection.connect()
        environments = get_json_config_default(machines, 'environments', {})
        replacements = get_json_config_default(machines, 'replacements', {})

        # 处理目录和文件混合的情况
        expanded_config_paths = []
        for config_path in self.args.tasks_config:
            if os.path.isdir(config_path):
                # 如果是目录，遍历目录下的所有JSON文件,用了if判断是否是json文件
                for root, _, files in os.walk(config_path):
                    for file in files:
                        if file.lower().endswith(('.json')) and os.path.join(root, file) not in expanded_config_paths:
                            expanded_config_paths.append(os.path.join(root, file))
            else:
                # 如果是文件，直接添加
                expanded_config_paths.append(config_path) if config_path not in expanded_config_paths else None

        for config_path in expanded_config_paths:
            is_valid_task_config, config=self.is_valid_task_config(config_path, replacements)
            if not is_valid_task_config:
                continue
            
            for name, task_config in config['tasks'].items():
                self.task_list.extend(self.parse_task(config, name, task_config, environments))    

    def merge_result(self):
        result_files = {}        
        for task in self.task_list:
            out_csv_list = task.output_manager.get_merged_result_info(self.global_logger)
            for one_csv in out_csv_list:
                file_name = one_csv[0]
                file_path = one_csv[1]
                if file_name in result_files.keys():
                    if file_path not in result_files[file_name]:
                        result_files[file_name].append(file_path)
                else:
                    result_files[file_name] = [file_path]

        for name, results in result_files.items():
            try:
                all_data = pd.DataFrame()
                for one_result in results:
                    data = pd.read_csv(one_result)
                    self.global_logger.info(f'merge result {one_result}, len={len(data)}')
                    all_data = pd.concat([all_data, data], ignore_index=True)

                output_path = os.path.join(self.now_path, f'merge_{name}')
                with open(output_path, 'w', encoding='utf-8') as csv_file:
                    all_data.to_csv(csv_file, index=False)
                self.global_logger.info(f"merge_{name} store in {output_path}, len={len(all_data)}")
            except Exception as e:
                self.global_logger.info(f"merge_result {name} exception {e}")

    def run(self):
        # 注册信号处理
        signal.signal(signal.SIGINT, self.handle_termination)   # 处理Ctrl+C
        signal.signal(signal.SIGTERM, self.handle_termination)  # 处理kill命令
        try:
            for index, task in enumerate(self.task_list):
                self.global_logger.info(f"==================== Task ({index + 1}/{len(self.task_list)}) {task.task_name} ====================")
                if self.finish_flag or self.is_stopped:
                    break
                task.set_real_progress_manager(self.real_progress_manager)
                self.current_task_ids = [index]
                task.run()
                self.current_task_ids = []

            if len(self.task_list) != 0:
                self.merge_result()
            self.connection.clean()
        except Exception as e:
            if len(self.current_task_ids) > 0:
                self.task_list[self.current_task_ids[0]].logger.exception(f'{e}')
            else:
                self.global_logger.exception(f'{e}')

    def run_single_task(self, task_index):
        try:
            if self.finish_flag or self.is_stopped:
                return
            task = self.task_list[task_index]
            self.global_logger.info(f'## ====== [{task_index}] Start Task [{task.task_name}] on {task.get_nodes_ips()} ======')
            task.set_real_progress_manager(self.real_progress_manager)
            with self.lock:
                self.current_task_ids.append(task_index)
            if self.is_stopped:
                return
            task.run()
        except Exception as e:
            task.logger.exception(f'{e}')
            task.stop_all(False)
        with self.lock:
            if task_index in self.current_task_ids:
                self.current_task_ids.remove(task_index)

    def run_parallel(self):
        if len(self.task_list) <= 1 or len(self.connection.nodes_info) == 1:
            self.run()
            return

        signal.signal(signal.SIGINT, self.handle_termination)   # 处理Ctrl+C
        signal.signal(signal.SIGTERM, self.handle_termination)  # 处理kill命令
        tasks_node_num = [task.get_node_num_need() for task in self.task_list]
        task_groups = self.solve_task_grouping(tasks_node_num, len(self.connection.nodes_info))
        for id, group in enumerate(task_groups):
            current_round = [f'{self.task_list[sub[0]].task_name}_server{self.task_list[sub[0]].task_id}_{sub[1]}nodes' for sub in group]
            self.global_logger.info(f'## Round {id}: {current_round}')
        try:
            with ThreadPoolExecutor(max_workers=len(self.connection.nodes_info)) as executor:
                for groups in task_groups:
                    current_group_futures = {}
                    current_nodes_sum = 0
                    for task_group in groups:
                        task_index, task_node_num = task_group[0], task_group[1]
                        self.task_list[task_index].set_node_used(current_nodes_sum, task_node_num)
                        future = executor.submit(self.run_single_task, task_index)
                        current_group_futures[future] = task_index
                        current_nodes_sum += task_node_num

                    for future in as_completed(current_group_futures):
                        task_index = current_group_futures[future]
                        self.global_logger.info(
                            f'## ====== [{task_index}] End Task [{self.task_list[task_index].task_name}] on {self.task_list[task_index].get_nodes_ips()} ======')
            if len(self.task_list) != 0:
                self.merge_result()
            self.connection.clean()
        except Exception as e:
            self.global_logger.exception(f'{e}')

    def solve_task_grouping(self, tasks_node_num, max_nodes_num):
        indexed_nums = list(enumerate(tasks_node_num))
        sorted_indexed_nums = sorted(indexed_nums, key=lambda x: x[1], reverse=True)
        groups = []
        current_group = []
        current_sum = 0

        for sorted_node in sorted_indexed_nums:
            node_num_need = sorted_node[1]
            assert (node_num_need <= max_nodes_num, f"task {sorted_node[0]} need {node_num_need} node, out of max {max_nodes_num}")
            if current_sum + node_num_need <= max_nodes_num:
                current_group.append(sorted_node)
                current_sum += node_num_need
            else:
                groups.append(current_group)
                current_group = [sorted_node]
                current_sum = node_num_need
                
        if current_group:
            groups.append(current_group)
        return groups

    def handle_termination(self, signal_num, frame):
        signal_name = signal.Signals(signal_num).name
        if self.is_stopped:
            sys.exit(0)
        self.global_logger.info(f"Recv SIG {signal_name}, Stop...")
        self.is_stopped = True
        try:
            with self.lock:
                # 停止当前任务
                for task_index in self.current_task_ids:
                    if isinstance(self.task_list[task_index], BaseTask):
                        try:
                            self.task_list[task_index].stop_all(False)
                            self.global_logger.info(f"Stop ({self.task_list[task_index].task_name}) success")
                        except Exception as e:
                            self.global_logger.info(f"Stop ({self.task_list[task_index].task_name}) failed: {e}")
                    
                self.current_task_ids = []
            # 清理连接
            self.connection.clean()
            
        except Exception as e:
            self.global_logger.exception(f"退出过程中发生错误: {str(e)}")
            sys.exit(1)  # 异常退出
        sys.exit(0) 

if __name__ == "__main__":
    benchmark_type_list = [
        BenchmarkType.perf.value,
        BenchmarkType.mmlu.value,
        BenchmarkType.ceval.value
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="output", help="Path for storing results")
    parser.add_argument("--tasks-config", type=str, nargs='*', help="JSON file or dir describing the task list")
    parser.add_argument("--machine-config", type=str, help="JSON file describing the machine list")
    parser.add_argument("--image-tag", type=str, default=" ",help="docker image tag")
    parser.add_argument('--specify-test', nargs='*', choices=benchmark_type_list, default=benchmark_type_list, help='special benchmark type to run, default all')
    parser.add_argument("--local-ip",type=str,required=True,help="default local ip")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    parser.add_argument("--timeout",type=int,default=1200,help="timeout in second for every task")
    # parser.add_argument('--parallel', action='store_true', help='run benchmark task parallel')
    
    Args = parser.parse_args(sys.argv[1:])

    task_scheduler = TaskScheduler(Args)
    task_scheduler.generate_task()
    # if Args.parallel:
    #     task_scheduler.run_parallel()
    # else:
    task_scheduler.run()
