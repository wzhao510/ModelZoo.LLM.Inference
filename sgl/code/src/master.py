import argparse
import sys
import os
# 获取当前文件所在目录的上一级目录（即项目的根目录）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

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
    def online_server(task, connect_mgr):
        # get total nodes's num and master machine's ip
        nodes_num = connect_mgr.get_nodes_num()
        master_ip = connect_mgr.get_master_machine_ip()

        # generate launch server's command list
        launch_server_commands = []
        for combo in product(*(task['launch_server'].values())):
            full_command = ""
            for param in combo:
                full_command += f" {param.strip()}" if full_command != "" else f"{param.strip()}"

            # put fields together
            full_command += f" --dist-init-addr %s:%s" % (master_ip, task['server_port'])
            full_command += f" --nnodes %s" % (str(nodes_num))
            full_command = full_command.strip()

            print(full_command)
            launch_server_commands.append(full_command)
        print("============================server command end=============================")
        return launch_server_commands
    
    @staticmethod
    def get_config_default(config, key, default_value):
        return config[key] if key in config.keys() else default_value
    
    @staticmethod
    def parse_benchmark(benchmark_config):
        command_base = ConfigParser.get_config_default(benchmark_config, 'command_base', '')
        benchmark_list = []
        for input_output in benchmark_config['input_output_len']:
            input_len, output_len = input_output.split('/')
            for bs in benchmark_config['num_prompt']:
                benchmark_list.append(
                    f" {command_base} --random-input-len {input_len} --random-output-len {output_len} --num-prompts {bs}"
                )
        return benchmark_list


class BenmchmarkParser(ConfigParser):
    @staticmethod
    # parse config and return TaskOffline/TaskONline list
    def from_config(config):
        pass


class PerfParser(ConfigParser):
    @staticmethod
    # parse config and return TaskOffline/TaskONline list
    def from_config(config):
        task_list = []
        connection = Connection(config['machine_info'], slave_port=20000)
        connection.connect()
        for task in config['tasks']:
            envs = ConfigParser.get_config_default(task, 'environment', [])
            envs['GLOO_SOCKET_IFNAME'] = connection.nodes_info[0].interface
            benchmark_list = ConfigParser.parse_benchmark(task['benchmark'])
            server_cmd = task['server_base']['command_base'] + ' '.join(task['server_base']['param'])
            launch_mode = ConfigParser.get_config_default(task, 'launch_mode', 'online')
            if launch_mode == 'online':
                task_list.append(TaskOnline(connection, server_cmd, benchmark_list, envs))
            else:
                for bench in benchmark_list:
                    task_list.append(TaskOffline(connection, server_cmd + ' ' + bench, envs))
        return task_list


class RampupParser(ConfigParser):
    @staticmethod
    def from_config(config):
        # parse config and return TaskOffline/TaskONline list
        # tasks = []

        # connections = []
        # environments = task['environment']
        # launch_server_cmds:list[str] = ConfigParser.online_server(config)

        # 填充爬坡的cmds
        config_benchmark = config['benchmark']
        bm_command_base = config_benchmark['command_base']
        bm_rampup_params = RampupParser.compose_rampup_params(config_benchmark)
        #variations
        bm_input_output_len_params = RampupParser.input_output_len(config_benchmark['input_output_len'])

        benchmark_cmds = []
        # 对所有input-output 适用rampup参数
        for bm_input_output_len_param in bm_input_output_len_params:
            for bm_rampup_param in bm_rampup_params:
                benchmark_cmd = bm_command_base + bm_rampup_param + bm_input_output_len_param
                benchmark_cmds.append(benchmark_cmd)

        # for lanuch_server_cmd in launch_server_cmds:
        #     task = TaskOnline(connections, lanuch_server_cmd, benchmark_cmds, environments)
        #     task.task_name = config['task_name']
        #     task.task_type = TaskType.rampup

        #     tasks.append(task)
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
                    rampup_param = f' --max-concurrency {max_concurrent_request} --warmup-requests {num_warmup_requests} --num-prompt {num_benchmark_requests} --ramp-up-period {rampup_period} --rampup-mode --flush-cache'
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
    
    @staticmethod
    def input_output_len(input_output_len_list):
        input_output_params = []
        for input_output_len in input_output_len_list:
            rnd_input_len, ran_output_len = input_output_len.split('/')
            input_output_param = f' --random-input-len {rnd_input_len} --random-output-len {ran_output_len}'
            input_output_params.append(input_output_param)

        return input_output_params

class TaskScheduler:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args:argparse.Namespace = args
        self.task_list = []
        self.current_task = None
        self.connection: Optional[Connection] = None
        self.finish_flag = False
        #...

    def stop(self):
        self.connection.clean()

    def parse_task_type(self, config) -> TaskType:
        if config.endswith("perf.json"):
            return TaskType.perf
        elif config.endswith("benchmark.json"):
            return TaskType.benchmark
        elif config.endswith("rampup_bench.json"):
            return TaskType.rampup

    def init_connection(self) -> None:
        config = read_json(self.args.tasks[0])
        self.connection = Connection(config['machine_info'], self.args.port)
        self.connection.connect()

    def init_output_manager(self) -> None:
        self.output_manager = OutputManager(self.args)
        self.output_manager.init_output_file(self.current_task)
        self.current_task.set_output_manager(self.output_manager)

    def generate_task(self) -> None:
        for config_path in self.args.tasks:
            config = read_json(config_path)
            task_type = self.parse_task_type(config_path)
            if task_type == TaskType.perf:
                self.task_list.extend(PerfParser.from_config(config))
            # elif task_type is TaskType.rampup:
            #     rampup_commands_list = RampupParser.from_config(task)
            #     for server in server_commands_list:
            #         self.task_list.append(TaskOnline(self.connection,
            #                                             server,
            #                                             rampup_commands_list,
            #                                             task['environment']))

    def get_run_task_id(self):
        pass

    def get_first_task(self):
        pass

    def get_next_task(self):
        pass

    def merge_result(self):
        pass

    def run(self):
        self.get_run_task_id()
        self.get_first_task()
        finish_flag = False
        # while not finish_flag:
        #     self.init_output_manager()
        #     self.current_task.run()
        #     self.get_next_task()
        for task in self.task_list:
            if self.finish_flag:
                break
            task.run()
        self.merge_result()
        self.stop()



# 异常停止处理


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
    # task_scheduler.init_connection()
    task_scheduler.generate_task()
    task_scheduler.run()
