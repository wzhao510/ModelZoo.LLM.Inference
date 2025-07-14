import argparse
import sys
import json
from typing import Optional, List, Dict, Any
from task import TaskOnline

from connection import Connection
from output import OutputManager

from utils.utils import *



class TaskScheduler:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args:argparse.Namespace = args
        self.task_list = []
        self.current_task = None
        self.json_data: Optional[Dict[str, Any]] = None
        self.connection: Optional[Connection] = None
        self.finish_flag = False
        #...

    def stop(self):
        self.connection.clean()

    def read_json(self) -> None:
        with open(self.args.json_file, 'r') as f:
            self.json_data = json.load(f)

    def init_connection(self) -> None:
        self.connection = Connection(self.json_data['machine_info'], self.args.port)
        self.connection.connect()

    def init_output_maneger(self) -> None:
        self.output_maneger = OutputManager(self.args)
        self.output_maneger.init_output_file(self.current_task)
        self.current_task.set_output_maneger(self.output_maneger)

    def generate_task(self) -> None:
        for task_config in self.json_data['tasks']:
            task_launch_mode = task['launch_mode']
            # parse param task type and create task from config
            # ...
            if task_launch_mode == TaskLaunchMode.online.value:
                task = TaskOnline.from_config(task_config)
                self.task_list.append(task)
                pass
            elif task_launch_mode == TaskLaunchMode.offline.value:
                pass
                # ...


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
        while not finish_flag:
            self.init_output_maneger()
            self.current_task.run()
            self.get_next_task()
        self.merge_result()
        self.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", type=str, default="/workspace", help="Target work path")
    parser.add_argument("--result-path", type=str, default="result", help="Path for storing results")
    parser.add_argument("--json-file", type=str, default="DeepSeek-R1-BF16/benchmark.json", help="JSON file describing the task list")
    parser.add_argument("--image_tag", type=str, default=" ",help="docker image tag")
    parser.add_argument("--progress-file", type=str, default="043.txt", help="real progress file")
    parser.add_argument("--pass-file", type=str, default="pass_file.txt", help="pass case id")
    parser.add_argument("--incremental-mode", action="store_true", help="only run case not in pass file")
    parser.add_argument("--specify-task", action="store_true", help="Starting from the designated task")
    parser.add_argument("--launchserver-id", type=int, default=0, help="Starting from launchserver-id")
    parser.add_argument("--benchserving-id", type=int, default=0, help="Starting benchserving-id")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    Args = parser.parse_args(sys.argv[1:])

    task_scheduler = TaskScheduler(Args)
    task_scheduler.read_json()
    task_scheduler.init_connection()
    task_scheduler.generate_task()
    task_scheduler.run()
