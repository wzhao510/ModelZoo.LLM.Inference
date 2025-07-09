
from typing import Optional, List

class BaseTask:
    def __init__(self) -> None:
        self.connection = None
        self.timer = None
        #...

    def run(self):
        pass

    def start_timer(self):
        pass

    def stop_timer(self):
        pass

    def pre_start_server(self):
        pass

    def start_server(self):
        pass

    def post_start_server(self):
        pass

    def stop_server(self):
        pass

    def bench_test(self):
        pass

    def generate_result(self):
        pass



class TaskOnline(BaseTask):
    def __init__(self) -> None:
        pass

    @staticmethod
    def from_config(config):
        # parse config and return list of TaskOnline
        return 


class TaskOffline(BaseTask):
    def __init__(self) -> None:
        pass

    @staticmethod
    def from_config(config):
        # parse config and return list of TaskOnline
        return 

