import argparse
from datetime import datetime
import os
from typing import Optional

from utils.utils import *


class OutputManager:

    def __init__(self,args:argparse.Namespace) -> None:
        self.args = args
        self.task = None
        self.log_file = None
        self.real_progress_file = None
        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")

    def init_output_file(self,task) -> None:
        self.task = task
        self.create_log_file()
        self.create_real_progress_file()
        configure_logger(log_file=self.log_file)

    def create_log_file(self) -> None:
        self.log_file = ""
        create_file(self.log_file)

    def create_real_progress_file(self) -> None:
        self.real_progress_file = ""
        create_file(self.real_progress_file)

    def create_pass_case_file(self) -> None:
        """ create pass case file"""
        file = os.path.join(self.args.target_path,self.args.result_path,self.args.pass_file)
        create_file(file)

    def create_test_result_file(self,task) -> None:
        result_file = self.get_client_result_file(task)
        create_file(result_file)


    def get_client_result_file(self,task) -> str:
        """ get online/offline bench/acc result file path"""
        if task.launch_mode == TaskLaunchMode.online:
            if task.task_type == TaskType.benchmark:
                pass
            elif task.task_type == TaskType.acc:
                pass
            elif task.task_type == TaskType.rampup:
                pass
            elif task.task_type == TaskType.perf:
                pass
            elif task.task_type == TaskType.search:
                pass
        return ''

    def write_client_result(self,file_path,result) -> None:
        """ write client result log to file"""
        pass

    def write_pass_case(self):
        """ write pass case id to file"""
        pass

    def write_next_task(self):
        """ write next case id to file"""
        pass

    def write_real_progress(self):
        """ write real progress to file"""
        pass

    def get_single_server_result_csv(self) -> None:
        """ get all client result csv of a server"""
        pass

    def fun(self):
        pass
