import argparse
from datetime import datetime
import os
from typing import Optional, TYPE_CHECKING, Union, List
import re
import pandas as pd
import json
import copy
import dataclasses

if TYPE_CHECKING:
    from src.task import TaskOnline, TaskOffline

from src.benchmark import Benchmark
from utils.utils import *

NOW_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclasses.dataclass
class OutputDepends:
    task_id: int = 0
    task_name: Optional[str] = ''
    launch_mode: Optional[str] = ''
    server_cmd: Optional[str] = ''
    server_full_cmd: Optional[str] = ''
    model_name: Optional[str] = ''
    node_id: int = 0
    output_path: Optional[str] = ''
    image_tag: Optional[str] = ''
    now_time: Optional[str] = ''

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))


class RealProgressManager:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args = args
        self.task = None
        self.task_full_name = ''
        self.fail_reason = 'fail'
        self.total_real_progress_path = os.path.join(self.args.output_path, f"total_real_progress_file.json")
        self.total_real_progress_data = {"to_run": "0,0", "docker_tag": self.args.image_tag, "tasks": list()}

        self.online_task_content = {}
        self.offline_task_content = {}
        self.now_time_path = os.path.join(self.args.output_path, NOW_TIME)

    def init(self, task):
        self.task = task
        self.task_full_name = f'{task.task_name.replace("-", "_")}_server{task.task_id}'
        self.online_task_content = {"launch_mode": "online",
                                    "simple_param": self.task_full_name,
                                    "server_id": f"{self.task.task_id}",
                                    "cmd": self.task.server_full_cmd,
                                    "client_test": list()}
        self.offline_task_content = {"launch_mode": "offline",
                                     "simple_param": self.task_full_name,
                                     "server_id": f"{self.task.task_id}",
                                     "cmd": self.task.server_full_cmd,
                                     "status": "",
                                     "times": 0}
        self.create_real_progress_file()
        self.write_real_progress_start()
        self.fail_reason = "Unknown Error"

    def create_real_progress_file(self) -> None:
        self.real_progress_data = {"to_run": "0,0", "docker_tag": self.args.image_tag, "tasks": list()}
        if not os.path.exists(self.total_real_progress_path):
            create_file(self.total_real_progress_path)

    def write_real_progress_start(self):
        # 因在线和离线的结构不同, 故在此处进行不同类型的处理
        if self.task.launch_mode == TaskLaunchMode.online:
            is_have_task = False
            for task in self.total_real_progress_data['tasks']:
                if task["server_id"] == f"{self.task.task_id}":
                    is_have_task = True
                    break
            if not is_have_task:
                self.total_real_progress_data['tasks'].append(self.online_task_content)
        elif self.task.launch_mode == TaskLaunchMode.offline:
            is_same = False
            for cur_task in self.total_real_progress_data['tasks']:
                if cur_task["server_id"] == f"{self.task.task_id}":
                    is_same = True
                    break
            if not is_same:
                self.total_real_progress_data['tasks'].append(self.offline_task_content)
            self.total_real_progress_data['to_run'] = "%s,%s" % (f"{self.task.task_id+1}", "0")

        self.write_real_progress_config()

    def write_real_progress_bench_serving(self, client_id, cmd, is_svr_start=True):
        id = self.task.task_id
        c_id = client_id + 1
        if not is_svr_start:
            id = self.task.task_id + 1
            c_id = client_id

        if (self.task.bench_total_num == client_id+1):
            self.total_real_progress_data['to_run'] = "%s,%s" % (str(self.task.task_id + 1), "0")
        else:
            self.total_real_progress_data['to_run'] = "%s,%s" % (str(id), str(c_id))

        if is_svr_start:
            self.total_real_progress_data = self.process_real_progress_bench_serving(self.total_real_progress_data, cmd, client_id)

        self.write_real_progress_config()

    def process_real_progress_bench_serving(self, content, cmd, client_id):
        for cur_task in content['tasks']:
            if cur_task["server_id"] != str(self.task.task_id):
                continue

            is_same = False
            for cur_client in cur_task["client_test"]:
                if cur_client["id"] == str(client_id):
                    is_same = True
                    break
            if not is_same:
                data = {"id": str(client_id),
                        "cmd": cmd,
                        "status": "",
                        "times": 0}
                cur_task["client_test"].append(copy.deepcopy(data))
            break
        return content

    def write_real_progress_result(self, result_flag, i):
        if self.task.launch_mode is TaskLaunchMode.online:
            self.total_real_progress_data = self.process_online_real_progress_result(result_flag, self.total_real_progress_data, i)
        elif self.task.launch_mode is TaskLaunchMode.offline:
            self.total_real_progress_data = self.process_offline_real_progress_result(result_flag, self.total_real_progress_data)

        self.write_real_progress_config()

    def process_online_real_progress_result(self, result_flag, data_content, i):
        for task_content in data_content["tasks"]:
            if task_content["server_id"] != str(self.task.task_id):
                continue
            for client_content in task_content["client_test"]:
                if client_content["id"] != str(i):
                    continue
                if result_flag == 'pass':
                    client_content["status"] = "pass"
                else:
                    client_content["status"] = self.fail_reason # 'fail'
                client_content["times"] += 1
                break
            break
        return data_content

    def process_offline_real_progress_result(self, result_flag, data_content):
        for task_content in data_content["tasks"]:
            if task_content["server_id"] != str(self.task.task_id):
                continue
            if result_flag == 'pass':
                task_content['status'] = "pass"
            else:
                task_content['status'] = self.fail_reason
            task_content['times'] += 1
            break
        return data_content

    def write_real_progress_config(self):
        if self.total_real_progress_path is None:
            return
        with open(self.total_real_progress_path, 'w') as f1:
            json.dump(self.total_real_progress_data, f1, indent=4, ensure_ascii=False)

    def get_total_real_progress_to_run(self):
        to_run_config = ["0", "0"]
        if not os.path.exists(self.total_real_progress_path):
            return to_run_config[0], to_run_config[1]

        with open(self.total_real_progress_path, "r+") as f:
            if os.path.getsize(self.total_real_progress_path) == 0:
                self.total_real_progress_data = {"to_run": "0,0", "docker_tag": self.args.image_tag, "tasks": []}
                json.dump(self.total_real_progress_data, f, indent=4, ensure_ascii=False)
            else:
                self.total_real_progress_data = json.load(f)
            to_run_config = self.total_real_progress_data['to_run'].split(",")

        return to_run_config[0], to_run_config[1]

    def write_to_run_args(self, task_id=0, client_id=0):
        self.total_real_progress_data['to_run'] = "%s,%s" % (str(task_id), str(client_id))
        self.write_real_progress_config()

    def set_fail_reason(self, reason):
        self.fail_reason = reason


class OutputManager:
    def __init__(self, output_depends: OutputDepends) -> None:
        self.output_depends = output_depends

        self.model_path = os.path.join(self.output_depends.output_path, self.output_depends.now_time, self.output_depends.model_name)
        self.task_full_name = f'{self.output_depends.task_name.replace("-", "_")}_server{self.output_depends.task_id}'

        self.log_path = os.path.join(self.model_path, "logs")
        self.log_file_name = f"{self.task_full_name}_node{self.output_depends.node_id}.log"
        self.log_file_path = os.path.join(self.log_path, self.log_file_name)
        self.result_path = os.path.join(self.model_path, "result", self.task_full_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.server_args = self._get_launch_server_args()
        self.logger = get_logger(self.log_path, self.log_file_name)
        
    @classmethod
    def from_task(cls, args:argparse.Namespace, task: Union["TaskOnline","TaskOffline"]):
        return cls(OutputDepends(
            task_id=task.task_id,
            task_name=task.task_name,
            launch_mode=task.launch_mode.value,
            server_cmd=task.server_cmd,
            model_name=task.model_name,
            node_id=0,
            output_path=args.output_path,
            image_tag=args.image_tag,
            now_time=NOW_TIME
        ))

    def _get_launch_server_args(self):
        command = self.output_depends.server_cmd
        server_args = {
                        'Build':[self.output_depends.image_tag],
                        'TaskName':[self.output_depends.task_name],
                        'ServerId':[str(self.output_depends.task_id)],
                        'Model':[self.output_depends.model_name],
                        'Torch Compile':[],
                        'Cache': [],
                        'chunked-prefix-cache':[],
                        'NextN': [],
                        'speculative-num-steps':[],
                        'speculative-eagle-topk':[],
                        'speculative-num-draft-tokens':[],
                        'Cuda Graph': [],
                        'flash_comm': [],
                        'Att Backend': [],
                        'dtype':[],
                        'Parallelism': [],
                        'cuda_graph_max_bs':[],
                        'chunked_prefill_size':[],
                        'mem_frac':[],
                        'embedding_tp_size':[],
                        'tp_size': []
                    }
        for key, value in server_args.items():
            if key == 'Cache':
                if re.search(r"--disable-radix-cache+", command):
                    value.append(None)
                else:
                    value.append("HiCache" if re.search(r"--enable-hierarchical-cache+", command) else "RadixCache")
            if key == 'chunked-prefix-cache':
                value.append('OFF' if re.search(r"--disable-chunked-prefix-cache+", command) else 'ON')
            elif key == 'Torch Compile':
                value.append("ON" if re.search(r"--enable-torch-compile+", command) else "OFF")
            elif key == 'NextN':
                value.append("ON" if re.search(r'--speculative-algorithm\s+NEXTN', command) else "OFF")

            elif key == 'speculative-num-steps':
                RE_match = re.search(r"--speculative-num-steps\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)
            elif key == 'speculative-eagle-topk':
                RE_match = re.search(r"--speculative-eagle-topk\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)
            elif key == 'speculative-num-draft-tokens':
                RE_match = re.search(r"--speculative-num-draft-tokens\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)

            elif key == 'Cuda Graph':
                value.append("OFF" if re.search(r"--disable-cuda-graph+", command) else "ON")
            elif key == 'flash_comm':
                value.append("ON" if re.search(r"--enable-flash-comm", command) else "OFF")
            elif key == 'Att Backend':
                if re.search(r"--attention-backend\s+(\S+)", command):
                    value.append(re.search(r"--attention-backend\s+(\S+)", command).group(1))
                elif re.search(r"--enable-flashmla", command):
                    value.append("flashmla")
                else:
                    value.append('default')

            elif key == 'dtype':
                if re.search(r"--dtype\s+(\S+)", command):
                    value.append(re.search(r"--dtype\s+(\S+)", command).group(1))
                else:
                    value.append('default')

            elif key == 'cuda_graph_max_bs':
                RE_match = re.search(r"--cuda-graph-max-bs\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)
            
            elif key == 'embedding_tp_size':
                RE_match = re.search(r"--embedding-tp-size\s+(\d+)", command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)

            elif key == 'chunked_prefill_size':
                RE_match = re.search(r"--chunked-prefill-size\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)

            elif key == 'mem_frac':
                RE_match = re.search(r"--mem-fraction-static\s+(\d+\.\d+|\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)
            

            elif key == 'Parallelism':
                para_str = ""
                tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
                ep_size_match = re.search(r"--(ep|ep-size)\s+(\d+)",command)
                dp_size_match = re.search(r"--(dp|dp-size)\s+(\d+)",command)
                pp_size_match = re.search(r"--(pp|pp-size)\s+(\d+)",command)

                tp_str = ''
                dp_str = ''
                ep_str = ''
                pp_str = ''

                assert tp_size_match
                tp_size = tp_size_match.group(2)
                tp_str = f"TP{tp_size}"

                server_args['tp_size'].append(tp_size)

                if dp_size_match:
                    dp_size = dp_size_match.group(2)
                    dp_str = f"DP{dp_size}" 
                    if re.search(r"--enable-dp-attention", command):
                        tp_str = f"TP{int(int(tp_size)/int(dp_size))}" 
  
                if pp_size_match:
                    pp_size = pp_size_match.group(2)
                    pp_str = f"PP{pp_size}"

                if ep_size_match:
                    ep_str = f"EP{ep_size_match.group(2)}"
                elif re.search(r"--enable-ep-moe", command):
                    ep_str = f"EP{tp_size}"

                value.append(f'{tp_str}{dp_str}{pp_str}{ep_str}')
        return server_args

    def append_result_file_param(self, command: str, benchmark: Benchmark) -> str:
        output_dir_param, output_file_param = benchmark.get_output_str(command)
        ret_cmd = command
        if output_dir_param != '':
            ret_cmd = f'{ret_cmd} {output_dir_param} {self.result_path}'
        if output_file_param != '':
            result_file_jsonl = os.path.join(
                self.result_path, f"{benchmark.get_benchmark_args_str(command)}.jsonl"
            )
            ret_cmd = f'{ret_cmd} {output_file_param} {result_file_jsonl}'
        return ret_cmd

    def write_client_result(self, command: str, benchmark: Benchmark, result, is_pass=True) -> None:
        result_file_text = os.path.join(
            self.result_path, f"{benchmark.get_benchmark_args_str(command)}.txt"
        )
        command_with_result = self.append_result_file_param(command, benchmark)
        if is_pass:
            with open(result_file_text, "w") as result_file:
                print(f"Command: {command_with_result}", file=result_file)
                print("\n".join(result), file=result_file)
        else:
            with open(result_file_text, "a") as result_file:
                try:
                    print("\n".join(result), file=result_file)
                except Exception as e:
                    self.logger.error(f"write_client_result exception {e}")
                pass

    def extract_result_metrics(
        self, 
        cmd: str,
        benchmark: Benchmark,
        running_server_args_str,
        running_server_args_dict,
    )-> None:
        original_dir = os.getcwd()
        try:
            os.chdir(self.result_path)
            result_file = f'{benchmark.get_benchmark_args_str(cmd)}{benchmark.get_result_ext()}'
            csv_file_name = f'{benchmark.get_csv_result_str()}_result.csv'
            if os.path.exists(csv_file_name):
                exist_result_df = pd.read_csv(csv_file_name)
            else:
                exist_result_df = None

            server_args = self.server_args
            if running_server_args_dict:
                server_args['mem_frac'][0] = running_server_args_dict['mem_fraction_static']
            
            if running_server_args_str:
                txt_file = f'{benchmark.get_benchmark_args_str(cmd)}.txt'
                with open(txt_file,'a') as f:
                    print(running_server_args_str, file=f)

            metrics = benchmark.extract_metrics_from_file(result_file)
            df_server_args = server_args.copy()
            df_server_args.pop('tp_size')
            server_args_df = pd.DataFrame(df_server_args)
            bench_result_df = pd.DataFrame(metrics)
            merge_df = pd.concat([server_args_df, bench_result_df], axis=1)
            if exist_result_df is not None:
                merge_df = pd.concat([exist_result_df, merge_df], ignore_index=True)

            with open(csv_file_name,'w',encoding='utf-8') as csv_file:
                merge_df.to_csv(csv_file, index=False)
                self.logger.debug(f"result_csv store in {csv_file_name}")
        finally:
            os.chdir(original_dir)

    def get_merged_result_info(self, logger):
        out_csv_list = []
        if not os.path.isdir(self.result_path):
            logger.info(f"result dir {self.result_path} not exist")
            return out_csv_list
        
        for file in os.listdir(self.result_path):
            file_path = os.path.join(self.result_path, file)
            if os.path.isfile(file_path) and file.endswith('.csv'):
                out_csv_list.append([file, file_path])

        return out_csv_list
