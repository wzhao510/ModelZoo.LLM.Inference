import argparse
import sys
#import os
# 获取当前文件所在目录的上一级目录（即项目的根目录）
#root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(root_dir)

import json
import re
from typing import Optional, List, Dict, Any
from src.task import TaskOnline, TaskOffline

from src.connection import Connection
from src.output import OutputManager

from utils.utils import *
from itertools import product


class ConfigParser:
    @staticmethod
    def online_server(server):
        # generate launch server's command list
        launch_server_commands = []
        for combo in product(*(server.values())):
            full_command = ""
            for param in combo:
                full_command += f" {param.strip()}" if full_command != "" else f"{param.strip()}"
            full_command = full_command.strip()

            print(full_command)
            launch_server_commands.append(full_command)
        print("============================server command end=============================")
        return launch_server_commands

    @staticmethod
    def get_config_default(config, key, default_value):
        return config[key] if key in config.keys() else default_value

    @staticmethod
    def parse_benchmark(benchmark_config, task_type):
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        benchmark_list = []
        for input_output in benchmark_config['input_output_len']:
            input_len, output_len = input_output.split('/')
            if task_type in [TaskType.benchmark,TaskType.perf]:
                for bs in benchmark_config['num_prompt']:
                    benchmark_list.append(
                        f" {command_base} --random-input-len {input_len} --random-output-len {output_len} --num-prompts {bs}"
                    )
            elif task_type == TaskType.search:
                  batch_size_config = benchmark_config["batch_size_config"]
                  bs_range = batch_size_config["range"]
                  steps = int(batch_size_config["steps"])
                  for bs in range(int(min(bs_range)),int(max(bs_range)),steps):
                    benchmark_list.append(
                            f" {command_base} --random-input-len {input_len} --random-output-len {output_len} --num-prompts {bs}"
                        )
        return benchmark_list


class BenmchmarkParser(ConfigParser):
    @staticmethod
    def from_config(config, connection, task_type):
        task_list = []

        for task in config['tasks']:
            server_list = ConfigParser.online_server(task['launch_server'])
            benchmark_list = ConfigParser.parse_benchmark(task['benchmark'], task_type)
            envs = ConfigParser.get_config_default(task, 'environment', [])
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            envs['MCCL_IB_HCA'] = connection.nodes_info[0].ib_hcas
            max_ttft = None
            max_tpot = None
            if task_type == TaskType.search:
                benchmark = task['benchmark']
                max_ttft = benchmark["max_ttft"]
                max_tpot = benchmark["max_tpot"]

            for server_cmd in server_list:
                _task = TaskOnline(connection, server_cmd, benchmark_list, envs, task['server_port'])
                _task.task_name = ConfigParser.get_config_default(task, 'task_name', f' ')
                _task.task_type = task_type
                _task.model_name = config['model_name']
                _task.max_ttft = max_ttft
                _task.max_tpot = max_tpot
                task_list.append(_task)

        return task_list


class PerfParser(ConfigParser):
    @staticmethod
    # parse config and return TaskOffline/TaskONline list
    def from_config(config, connection,task_type):
        task_list = []
        task_id = -1
        for task in config['tasks']:
            envs = ConfigParser.get_config_default(task, 'environment', [])
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            envs['MCCL_IB_HCA'] = connection.nodes_info[0].ib_hcas
            benchmark_list = ConfigParser.parse_benchmark(task['benchmark'],task_type)
            server_cmd = task['server_base']['command_base']+" " + ' '.join(task['server_base']['param'])
            launch_mode = ConfigParser.get_config_default(task, 'launch_mode', 'online')

            if launch_mode == 'online':
                _task = TaskOnline(connection, server_cmd, benchmark_list, envs, task['server_port'])
                _task.task_name = task['task_name']
                _task.task_type = TaskType.perf
                _task.model_name = config['model_name']
                task_list.append(_task)
            else:
                for benchmark in benchmark_list:
                    task_id += 1
                    _task = TaskOffline(connection, server_cmd+" "+benchmark ,task_id, envs, task['server_port'])
                    _task.task_name = task['task_name']
                    _task.task_type = TaskType.perf
                    _task.model_name = config['model_name']
                    task_list.append(_task)
        return task_list


class RampupParser(ConfigParser):
    @staticmethod
    def from_config(config, connection):
        # parse config and return TaskOffline/TaskONline list
        task_list = []

        for task_id, task in enumerate(config['tasks']):
            server_cmds:list[str] = ConfigParser.online_server(task['launch_server'])
            benchmark_cmds = RampupParser.get_rampup_benchmark_cmds(task['benchmark'])
            envs = ConfigParser.get_config_default(task, 'environment', {})
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface

            for cmd_id, server_cmd in enumerate(server_cmds):
                _task = TaskOnline(connection, server_cmd, benchmark_cmds, envs, task['server_port'])
                _task.task_name = ConfigParser.get_config_default(task, 'task_name', f'task{task_id}_{cmd_id}')
                _task.task_type = TaskType.rampup
                _task.model_name = config['model_name']

                task_list.append(_task)

        return task_list
        
    @staticmethod
    def get_rampup_benchmark_cmds(benchmark_config):
        # 填充爬坡的cmds
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        rampup_params = RampupParser.compose_rampup_params(benchmark_config)
        #variations
        # bm_input_output_len_params = RampupParser.input_output_len(benchmark_config['input_output_len'])

        benchmark_cmds = []
        # 对所有input-output 适用rampup参数
        for input_output in benchmark_config['input_output_len']:
            input_len, output_len = input_output.split('/')
            for rampup_param in rampup_params:
                benchmark_cmds.append(
                    f" {command_base} {rampup_param} --random-input-len {input_len} --random-output-len {output_len}"
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
        #" --least_requests_num 16 --num_warmup_requests_ratio 4 --num_benchmark_requests_ratio 16 "
        pattern = r'--(\w+)\s+(\d+)'
        matches = re.findall(pattern, requests_config)
        params = dict(matches)
        
        # 提取目标参数（如果不存在则返回 None）
        least_reqs_num = int(params.get('least_requests_num'))
        warmup_reqs_ratio = int(params.get('num_warmup_requests_ratio'))
        benchmark_reqs_ratio = int(params.get('num_benchmark_requests_ratio'))

        return least_reqs_num, warmup_reqs_ratio, benchmark_reqs_ratio
    
class AccParser(ConfigParser):
    @staticmethod
    def from_config(config, connection):
        task_list = []

        for task_id, task in enumerate(config['tasks']):
            server_cmds = ConfigParser.online_server(task['launch_server'])
            acc_type = TaskAccType(ConfigParser.get_config_default(task, 'acc_type', TaskAccType.mmlu.value))
            benchmark_cmds = AccParser.get_acc_benchmark_cmds(acc_type, task['benchmark'])
            envs = ConfigParser.get_config_default(task, 'environment', {})
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface

            for cmd_id, server_cmd in enumerate(server_cmds):
                _task = TaskOnline(connection, server_cmd, benchmark_cmds, envs, task['server_port'])
                _task.task_name = ConfigParser.get_config_default(task, 'task_name', f'task{task_id}_{cmd_id}')
                _task.task_type = TaskType.acc
                _task.acc_type = acc_type
                _task.model_name = config['model_name']

                task_list.append(_task)

        return task_list
    
    @staticmethod
    def get_acc_benchmark_cmds(acc_type:TaskAccType, benchmark_config):
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        if acc_type == TaskAccType.mmlu:
            return [command_base]
        
        elif acc_type == TaskAccType.ceval:
            benchmark_cmds = []
            for random in benchmark_config['random']:
                benchmark_cmds.append(
                    f"{command_base} {random}"
                )
            return benchmark_cmds

class TaskScheduler:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args:argparse.Namespace = args
        self.task_list = []
        self.current_task = None
        self.finish_flag = False
        self.connection:Optional[Connection] = None

    def parse_task_type(self, config) -> TaskType:
        if config.endswith("perf.json"):
            return TaskType.perf
        elif config.endswith("benchmark.json"):
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

    def init_output_manager(self) -> None:
        # self.output_manager = OutputManager(self.args)
        # self.output_manager.init_output_file(self.current_task)
        # self.current_task.set_output_manager(self.output_manager)
        pass

    def generate_task(self) -> None:
        for config_path in self.args.tasks:
            config = read_json(config_path)
            task_type = self.parse_task_type(config_path)
            config['model_name'] = self.extract_task_model_name(config_path)
            self.connection = Connection(config['machine_info'], self.args.port)
            self.connection.connect()
            if task_type == TaskType.perf:
                self.task_list.extend(PerfParser.from_config(config, self.connection,task_type))
            elif task_type == TaskType.rampup:
                self.task_list.extend(RampupParser.from_config(config, self.connection))
            elif task_type == TaskType.benchmark or task_type == TaskType.search:
                self.task_list.extend(BenmchmarkParser.from_config(config, self.connection, task_type))
            elif task_type == TaskType.acc:
                self.task_list.extend(AccParser.from_config(config, self.connection))

    def get_run_task_id(self):
        pass

    def get_first_task(self):
        pass

    def get_next_task(self):
        pass

    def merge_result(self, output_manager:OutputManager):
        # online
        online_tasks = [task for task in self.task_list if task.launch_mode == TaskLaunchMode.online]
        task_types = [task.task_type for task in online_tasks]
        task_types = list(dict.fromkeys(task_types))    # 去重
        for task_type in task_types:
            if task_type == TaskType.search:
                output_manager.merge_online_search_result(task_type)
            else:
                output_manager.merge_online_result(task_type)

        # offline
        offline_tasks = [task for task in self.task_list if task.launch_mode == TaskLaunchMode.offline]
        task_types = [task.task_type for task in offline_tasks]
        task_types = list(dict.fromkeys(task_types))    # 去重
        for task_type in task_types:
            output_manager.merge_offline_result(task_type)

    def run(self):
        self.get_run_task_id()
        self.get_first_task()
        finish_flag = False
        # while not finish_flag:
        #     self.init_output_manager()
        #     self.current_task.run()
        #     self.get_next_task()
        output_manager = OutputManager(self.args)
        output_manager.set_all_task_nums(len(self.task_list))
        for task in self.task_list:
            if self.finish_flag:
                break
            task.set_output_manager(output_manager)
            task.run()
        self.merge_result(output_manager)
        self.connection.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="output", help="Path for storing results")
    parser.add_argument("--tasks", type=str,nargs='*', default=["DeepSeek-R1-BF16/benchmark.json"], help="JSON file describing the task list")
    parser.add_argument("--image-tag", type=str, default=" ",help="docker image tag")
    parser.add_argument("--incremental-mode", action="store_true", help="only run case not in pass file")
    parser.add_argument("--specify-task", action="store_true", help="Starting from the designated task")
    parser.add_argument("--launchserver-id", type=int, default=0, help="Starting from launchserver-id")
    parser.add_argument("--benchserving-id", type=int, default=0, help="Starting benchserving-id")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    Args = parser.parse_args(sys.argv[1:])

    task_scheduler = TaskScheduler(Args)
    task_scheduler.generate_task()
    task_scheduler.run()
