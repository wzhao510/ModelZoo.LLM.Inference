import argparse
import sys
import signal
#import os
# 获取当前文件所在目录的上一级目录（即项目的根目录）
#root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(root_dir)

import json
import re
from typing import Optional, List, Dict, Any, Union
from src.task import TaskOnline, TaskOffline,BaseTask

from src.connection import Connection
from src.output import BenchmarkOutputManager, AccOutputManager, SearchOutputManager, RampupOutputManager, RealProgressManager

from utils.utils import *
from itertools import product
from collections import defaultdict
import pandas as pd


server_id_global = -1
server_pass_list = defaultdict(list)


class ConfigParser:
    @staticmethod
    def online_server(server, logger = None):
        # generate launch server's command list
        launch_server_commands = []
        for combo in product(*(server.values())):
            full_command = ""
            for param in combo:
                full_command += f" {param.strip()}" if full_command != "" else f"{param.strip()}"
            full_command = full_command.strip()

            log_msg_level(full_command, logger)
            launch_server_commands.append(full_command)
        return launch_server_commands

    @staticmethod
    def get_config_default(config, key, default_value):
        return config[key] if key in config.keys() else default_value

    @staticmethod
    def parse_benchmark(benchmark_config, task_type, launch_mode):
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '') if launch_mode == 'online' else ''
        client_id = -1
        benchmark_list = []
        for input_output in benchmark_config['input_output_len']:
            input_len, output_len = input_output.split('/')
            if task_type in [TaskType.benchmark]:
                for bs in benchmark_config['num_prompt']:
                    client_id += 1
                    benchmark_list.append(
                        BenchmarkCmds(client_id, f" {command_base} --random-input-len {input_len} --random-output-len {output_len} --num-prompts {bs}")
                    )
            elif task_type == TaskType.search:
                batch_size_config = benchmark_config["batch_size_config"]
                bs_range = batch_size_config["range"]
                steps = int(batch_size_config["steps"])
                for bs in range(int(min(bs_range)),int(max(bs_range)),steps):
                    client_id += 1
                    benchmark_list.append(
                        BenchmarkCmds(client_id, f" {command_base} --random-input-len {input_len} --random-output-len {output_len} --num-prompts {bs}")
                    )
        return benchmark_list

    @staticmethod
    def filter_benchmark(svr_id, benchmark_list, incremental_mode):
        global server_pass_list

        if incremental_mode:
            benchmark_list_filter = []
            for index, item in enumerate(benchmark_list):
                if str(index) in server_pass_list[str(svr_id)]:
                    continue
                benchmark_list_filter.append(item)
            return benchmark_list_filter
        return benchmark_list
        

class BenmchmarkParser(ConfigParser):
    @staticmethod
    def from_config(config, connection, task_type, incremental_mode, args, common_envs, logger = None):
        task_list = []
        global server_id_global

        for task in config['tasks']:
            server_list = ConfigParser.online_server(task['launch_server'], logger=logger)
            launch_mode = ConfigParser.get_config_default(task, 'launch_mode', 'online')
            benchmark_list = ConfigParser.parse_benchmark(task['benchmark'], task_type, launch_mode)
            envs = common_envs.copy()
            envs.update(ConfigParser.get_config_default(task, 'environment', {}))
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            envs['MCCL_IB_HCA'] = connection.nodes_info[0].ib_hcas
            max_ttft = None
            max_tpot = None
            if task_type == TaskType.search:
                benchmark = task['benchmark']
                max_ttft = benchmark["max_ttft"]
                max_tpot = benchmark["max_tpot"]

            for server_cmd in server_list:
                if launch_mode == 'online':
                    server_id_global += 1
                    if str(server_id_global) in server_pass_list and len(server_pass_list[str(server_id_global)]) == len(benchmark_list):
                        continue

                    benchmark_list_filter = ConfigParser.filter_benchmark(server_id_global, benchmark_list, incremental_mode)
                    _task = TaskOnline(connection, server_cmd, benchmark_list_filter, server_id_global, envs, task['server_port'])
                    _task.task_name = ConfigParser.get_config_default(task, 'task_name', f' ')
                    _task.task_type = task_type
                    _task.model_name = config['model_name']
                    _task.max_ttft = max_ttft
                    _task.max_tpot = max_tpot
                    if task_type == TaskType.search:
                        _task.set_output_manager(SearchOutputManager(args, _task))
                    else:
                        _task.set_output_manager(BenchmarkOutputManager(args, _task))
                    _task.global_logger = logger
                    task_list.append(_task)
                else:
                    for benchmark in benchmark_list:
                        server_id_global += 1
                        if str(server_id_global) in server_pass_list:
                            continue
                        _task = TaskOffline(connection, server_cmd+" "+benchmark.get_cmd(), server_id_global, envs, task['server_port'])
                        _task.task_name = task['task_name']
                        _task.task_type = task_type
                        _task.model_name = config['model_name']
                        if task_type == TaskType.search:
                            _task.set_output_manager(SearchOutputManager(args, _task))
                        else:
                            _task.set_output_manager(BenchmarkOutputManager(args, _task))
                        _task.global_logger = logger
                        task_list.append(_task)
        log_msg_level(f'Benmchmark task list len={len(task_list)}', logger)
        return task_list


class RampupParser(ConfigParser):
    @staticmethod
    def from_config(config, connection, incremental_mode, args):
        global server_id_global
        # parse config and return TaskOffline/TaskONline list
        task_list = []

        for task_id, task in enumerate(config['tasks']):
            server_cmds:list[str] = ConfigParser.online_server(task['launch_server'])
            benchmark_cmds = RampupParser.get_rampup_benchmark_cmds(task['benchmark'])
            envs = ConfigParser.get_config_default(task, 'environment', {})
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            envs['MCCL_IB_HCA'] = connection.nodes_info[0].ib_hcas

            for cmd_id, server_cmd in enumerate(server_cmds):
                server_id_global += 1
                if str(server_id_global) in server_pass_list and len(server_pass_list[str(server_id_global)]) == len(benchmark_cmds):
                    continue
                benchmark_list_filter = ConfigParser.filter_benchmark(server_id_global, benchmark_cmds, incremental_mode)
                _task = TaskOnline(connection, server_cmd, benchmark_list_filter, server_id_global, envs, task['server_port'])
                _task.task_name = ConfigParser.get_config_default(task, 'task_name', f'task{task_id}_{cmd_id}')
                _task.task_type = TaskType.rampup
                _task.model_name = config['model_name']

                _task.set_output_manager(RampupOutputManager(args, _task))
                task_list.append(_task)

        return task_list

    @staticmethod
    def get_rampup_benchmark_cmds(benchmark_config):
        # 填充爬坡的cmds
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        rampup_params = RampupParser.compose_rampup_params(benchmark_config)
        #variations
        # bm_input_output_len_params = RampupParser.input_output_len(benchmark_config['input_output_len'])

        client_id = -1
        benchmark_cmds = []
        # 对所有input-output 适用rampup参数
        for input_output in benchmark_config['input_output_len']:
            input_len, output_len = input_output.split('/')
            for rampup_param in rampup_params:
                client_id += 1
                benchmark_cmds.append(
                    BenchmarkCmds(client_id, f" {command_base} {rampup_param} --random-input-len {input_len} --random-output-len {output_len}")
                )
        return benchmark_cmds

    @staticmethod
    def compose_rampup_params(config_benchmark):
        max_concurrent_requests:list[str] = config_benchmark['max_concurrent_requests'] #暂时认为他是单一的
        rampup_period_list:list[str] = config_benchmark['rampup_period'] #多组
        requests_config_list:list[str] = config_benchmark['requests_configs']

        max_concurrent_requests = [int(x) for x in max_concurrent_requests]
        
        # 使用上面的参数按爬坡的顺序组合
        rampup_params = []
        for requests_config in requests_config_list:
            least_requests_num, num_warmup_requests_ratio, num_benchmark_requests_ratio = RampupParser.parse_request_configs(requests_config)
            for rampup_periods in rampup_period_list:
                rampup_periods = [int(x) for x in rampup_periods.split(',')]
        
                for max_concurrent_request, rampup_period in zip(max_concurrent_requests,rampup_periods):
                    num_warmup_requests = max(least_requests_num, int(max_concurrent_request)*num_warmup_requests_ratio)
                    num_benchmark_requests = max(least_requests_num, int(max_concurrent_request)*num_benchmark_requests_ratio)
                    rampup_param = f' --max-concurrency {max_concurrent_request} --warmup-requests {num_warmup_requests} --num-prompts {num_benchmark_requests} --ramp-up-period {rampup_period} --rampup-mode --flush-cache'
                    rampup_params.append(rampup_param)

        return rampup_params

    @staticmethod
    def parse_request_configs(requests_config:str):
        """
        --least_requests_num 16 --num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16
        或
        --num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16
        """
        pattern = r'--(\w+)\s+(\d+)'
        matches = re.findall(pattern, requests_config)
        params = dict(matches)
        
        # 提取目标参数（如果least_requests_num不存在则返回 16)
        least_reqs_num = int(params.get('least_requests_num', 16))
        warmup_reqs_ratio = int(params.get('num_warmup_requests_ratio'))
        benchmark_reqs_ratio = int(params.get('num_benchmark_requests_ratio'))

        return least_reqs_num, warmup_reqs_ratio, benchmark_reqs_ratio


class AccParser(ConfigParser):
    @staticmethod
    def from_config(config, connection, incremental_mode, args, common_envs, logger = None):
        global server_id_global
        task_list = []

        for task_id, task in enumerate(config['tasks']):
            server_cmds = ConfigParser.online_server(task['launch_server'], logger=logger)
            acc_type = TaskAccType(ConfigParser.get_config_default(task, 'acc_type', TaskAccType.mmlu.value))
            benchmark_cmds = AccParser.get_acc_benchmark_cmds(acc_type, task['benchmark'])
            envs = common_envs.copy()
            envs.update(ConfigParser.get_config_default(task, 'environment', {}))
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            envs['MCCL_IB_HCA'] = connection.nodes_info[0].ib_hcas

            for cmd_id, server_cmd in enumerate(server_cmds):
                server_id_global += 1
                if str(server_id_global) in server_pass_list and len(server_pass_list[str(server_id_global)]) == len(benchmark_cmds):
                    continue
                benchmark_list_filter = ConfigParser.filter_benchmark(server_id_global, benchmark_cmds, incremental_mode)
                _task = TaskOnline(connection, server_cmd, benchmark_list_filter, server_id_global, envs, task['server_port'])
                _task.task_name = ConfigParser.get_config_default(task, 'task_name', f'task{task_id}_{cmd_id}')
                _task.task_type = TaskType.acc
                _task.acc_type = acc_type
                _task.model_name = config['model_name']

                _task.set_output_manager(AccOutputManager(args, _task))
                _task.global_logger = logger
                task_list.append(_task)
        log_msg_level(f'Acc task list len={len(task_list)}', logger)
        return task_list

    @staticmethod
    def get_acc_benchmark_cmds(acc_type:TaskAccType, benchmark_config):
        client_id = -1

        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        if acc_type == TaskAccType.mmlu:
            client_id += 1
            return [BenchmarkCmds(client_id, command_base)]

        elif acc_type == TaskAccType.ceval:
            benchmark_cmds = []
            if 'random' in benchmark_config:
                for random in benchmark_config['random']:
                    client_id += 1
                    benchmark_cmds.append(
                        BenchmarkCmds(client_id, f"{command_base} {random}")
                    )
            else:
                client_id += 1
                benchmark_cmds.append(
                    BenchmarkCmds(client_id, f"{command_base}")
                )
            return benchmark_cmds


class TaskScheduler:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args:argparse.Namespace = args
        self.task_list = []
        self.current_task = None
        self.finish_flag = False
        self.connection:Optional[Connection] = None
        self.is_stopped = False
        self.real_progress_manager = RealProgressManager(self.args)
        self.now_path = self.real_progress_manager.now_time_path
        self.global_logger = get_logger(self.now_path, 'bench_record.log')

    def parse_task_type(self, config) -> TaskType:
        if config.endswith("benchmark.json"):
            return TaskType.benchmark
        elif config.endswith("rampup_bench.json"):
            return TaskType.rampup
        elif config.endswith('acc.json'):
            return TaskType.acc
        elif config.endswith("search.json"):
            return TaskType.search
        
    def extract_task_model_name(self, config_path) -> str:
        # 默认认为倒数第一个路径是模型名
        return config_path.split('/')[-2]

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

        global server_pass_list
        for task in task_list:
            if task["launch_mode"] == "online":
                for client in task["client_test"]:
                    if client["status"] != "pass":
                        continue
                    server_pass_list[task['server_id']].append(client["id"])
            elif task["launch_mode"] == "offline":
                if task["status"] != "pass":
                    continue
                server_pass_list[task['server_id']] = []
            else:
                self.global_logger.error(f'launch_mode is unknown type...' + task["launch_mode"] + " server id: " + task["server_id"])
        self.global_logger.info("current server_pass_list after filter:")
        self.global_logger.info(server_pass_list)

    def generate_task(self) -> None:
        # 跑非PASS测试, 需要设置此参数
        # 此处将pass的测试id添加到全局变量 server_pass_list, 后续筛选使用
        # if self.args.incremental_mode:
        #     self.pass_id_filter()
        incremental_mode = False
        machines = read_json(self.args.machine_config)
        self.connection = Connection(machines['machine_info'], self.args.port, logger=self.global_logger)
        self.connection.connect()
        common_envs = ConfigParser.get_config_default(machines, 'common_envs', {})

        for config_path in self.args.tasks_config:
            config = read_json(config_path)
            task_type = self.parse_task_type(config_path)
            config['model_name'] = self.extract_task_model_name(config_path)
            # if task_type == TaskType.rampup:
            #     self.task_list.extend(RampupParser.from_config(config, self.connection, self.args.incremental_mode, self.args))
            if task_type == TaskType.benchmark: # or task_type == TaskType.search:
                self.task_list.extend(BenmchmarkParser.from_config(
                    config, self.connection, task_type, incremental_mode, self.args, common_envs, logger=self.global_logger))
            elif task_type == TaskType.acc:
                self.task_list.extend(AccParser.from_config(
                    config, self.connection, incremental_mode, self.args, common_envs, logger=self.global_logger))
                

    def merge_result(self):
        task_types = [task.task_type for task in self.task_list]
        task_types = list(dict.fromkeys(task_types))    # 去重
        
        for task_type in task_types:
            try:
                if task_type == TaskType.search:
                    search_tasks = [task for task in self.task_list if task.task_type == TaskType.search]
                    search_task = search_tasks[0]
                    search_task.output_manager.merge_online_search_result(task_type,search_task.max_ttft, search_task.max_tpot)
                else:
                    path_merged = []
                    all_data = pd.DataFrame()
                    for task in self.task_list:
                        if task.task_type == task_type:
                            data = task.output_manager.merge_result(path_merged, self.global_logger)
                            all_data = pd.concat([all_data, data], ignore_index=True)
                    output_path = os.path.join(self.now_path, f'{task_type.value}.csv')
                    with open(output_path, 'w', encoding='utf-8') as csv_file:
                        all_data.to_csv(csv_file, index=False)
                    self.global_logger.info(f"{task_type.value} result_csv store in {output_path}, len={len(all_data)}")
            except Exception as e:
                self.global_logger.info(f"merge_result {task_type.value} exception {e}")

    def run(self):
        # 注册信号处理
        signal.signal(signal.SIGINT, self.handle_termination)   # 处理Ctrl+C
        signal.signal(signal.SIGTERM, self.handle_termination)  # 处理kill命令
        try:
            to_run_task_id = "0"
            client_id = "0"
            # if self.args.specify_task:
            #     to_run_task_id, client_id = real_progress_manager.get_total_real_progress_to_run()

            is_task_skip = True
            for index, task in enumerate(self.task_list):
                self.global_logger.info(f"==================== Task ({index + 1}/{len(self.task_list)}) {task.task_name} ====================")
                if self.finish_flag or self.is_stopped:
                    break

                if index < (len(self.task_list) - 1):
                    task.set_is_kill_abnormal(False)
                else:
                    task.set_is_kill_abnormal(True)

                # 处理进程异常退出的情况
                # 1.默认0,0  从第0个server第0个bench开始run
                # 2.后续x,y  从第x个server第y个bench开始run
                if is_task_skip:
                    self.global_logger.info("current task need skip task id " + to_run_task_id + " current id " + str(task.task_id))
                    if task.task_id < int(to_run_task_id):
                        continue
                    is_task_skip = False
                    self.global_logger.info("current task need skip done.." )
                task.output_manager.init()
                # real_progress_manager.init(task)
                task.set_real_progress_manager(self.real_progress_manager)
                self.current_task = task
                task.run(int(client_id))
                # 此处需要再次初始化,防止跳过其他task的bench
                client_id = "0"
                self.current_task = None

            if len(self.task_list) != 0:
                self.real_progress_manager.write_to_run_args()
                self.merge_result()
            self.connection.clean()
        except Exception as e:
            if self.current_task is not None:
                self.current_task.logger.exception(f'{e}')
            else:
                self.global_logger.exception(f'{e}')

    def handle_termination(self, signal_num, frame):
        signal_name = signal.Signals(signal_num).name
        if self.is_stopped:
            sys.exit(0)
        self.global_logger.info(f"Recv SIG {signal_name}, Stop...")
        self.is_stopped = True
        try:
            # 停止当前任务
            if isinstance(self.current_task, BaseTask):
                try:
                    self.current_task.stop_all(False)
                    self.global_logger.info(f"Stop ({type(self.current_task).__name__}) success")
                except Exception as e:
                    self.global_logger.info(f"Stop ({type(self.current_task).__name__}) failed: {e}")
                
            # 清理连接
            self.connection.clean()
            
        except Exception as e:
            self.global_logger.exception(f"退出过程中发生错误: {str(e)}")
            sys.exit(1)  # 异常退出
        sys.exit(0) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="output", help="Path for storing results")
    parser.add_argument("--tasks-config", type=str, nargs='*', help="JSON file describing the task list")
    parser.add_argument("--machine-config", type=str, help="JSON file describing the machine list")
    parser.add_argument("--image-tag", type=str, default=" ",help="docker image tag")
    # parser.add_argument("--incremental-mode", action="store_true", help="only run case not in pass file")
    # parser.add_argument("--specify-task", action="store_true", help="Starting from the designated task")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    Args = parser.parse_args(sys.argv[1:])

    task_scheduler = TaskScheduler(Args)
    task_scheduler.generate_task()
    task_scheduler.run()
