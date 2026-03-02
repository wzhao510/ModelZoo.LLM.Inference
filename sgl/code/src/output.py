import argparse
from datetime import datetime
import os
from typing import Optional, TYPE_CHECKING, Union, List
import re
import pandas as pd
import json
import copy
import dataclasses
import threading

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
    node_ip: Optional[str] = ''
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
        self.total_real_progress_path = os.path.join(self.args.output_path, f"total_real_progress_file.json")
        self.total_real_progress_data = {"docker_tag": self.args.image_tag, "tasks": list()}
        self.now_time_path = os.path.join(self.args.output_path, NOW_TIME)
        self.lock = threading.Lock()
        if not os.path.exists(self.total_real_progress_path):
            create_file(self.total_real_progress_path)

    def init_task_content(self, task):
        task_full_name = f'{task.task_name.replace("-", "_")}_server{task.task_id}'
        online_task_content = {"launch_mode": "online",
                                    "simple_param": task_full_name,
                                    "server_id": f"{task.task_id}",
                                    "cmd": task.server_full_cmd,
                                    "client_test": list()}
        offline_task_content = {"launch_mode": "offline",
                                     "simple_param": task_full_name,
                                     "server_id": f"{task.task_id}",
                                     "cmd": task.server_full_cmd,
                                     "status": "",
                                     "times": 0}
        with self.lock:
            # 因在线和离线的结构不同, 故在此处进行不同类型的处理
            if task.launch_mode == TaskLaunchMode.online:
                is_have_task = False
                for task_content in self.total_real_progress_data['tasks']:
                    if task_content["server_id"] == f"{task.task_id}":
                        is_have_task = True
                        break
                if not is_have_task:
                    self.total_real_progress_data['tasks'].append(online_task_content)
            elif task.launch_mode == TaskLaunchMode.offline:
                is_same = False
                for task_content in self.total_real_progress_data['tasks']:
                    if task_content["server_id"] == f"{task.task_id}":
                        is_same = True
                        break
                if not is_same:
                    self.total_real_progress_data['tasks'].append(offline_task_content)
            self._write_real_progress_config()

    def write_real_progress_bench_serving(self, task_id, client_id, cmd):
        with self.lock:
            for cur_task in self.total_real_progress_data['tasks']:
                if cur_task["server_id"] != str(task_id):
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
            self._write_real_progress_config()

    def write_real_progress_result(self, result_flag, task, client_id = 0):
        with self.lock:
            if task.launch_mode is TaskLaunchMode.online:
                self._process_online_real_progress_result(result_flag, task, client_id)
            elif task.launch_mode is TaskLaunchMode.offline:
                self._process_offline_real_progress_result(result_flag, task)

            self._write_real_progress_config()

    def _process_online_real_progress_result(self, result_flag, task, client_id):
        for task_content in self.total_real_progress_data["tasks"]:
            if task_content["server_id"] != str(task.task_id):
                continue
            for client_content in task_content["client_test"]:
                if client_content["id"] != str(client_id):
                    continue
                if result_flag == 'pass':
                    client_content["status"] = "pass"
                else:
                    client_content["status"] = task.failed_reason
                client_content["times"] += 1
                break
            break

    def _process_offline_real_progress_result(self, result_flag, task):
        for task_content in self.total_real_progress_data["tasks"]:
            if task_content["server_id"] != str(task.task_id):
                continue
            if result_flag == 'pass':
                task_content['status'] = "pass"
            else:
                task_content['status'] = task.failed_reason
            task_content['times'] += 1
            break

    def _write_real_progress_config(self):
        if self.total_real_progress_path is None:
            return
        with open(self.total_real_progress_path, 'w') as f1:
            json.dump(self.total_real_progress_data, f1, indent=4, ensure_ascii=False)


class OutputManager:
    def __init__(self, output_depends: OutputDepends) -> None:
        self.output_depends = output_depends

        self.model_path = os.path.join(self.output_depends.output_path, self.output_depends.now_time, self.output_depends.model_name)
        self.task_full_name = f'{self.output_depends.task_name.replace("-", "_")}_server{self.output_depends.task_id}'

        self.log_path = os.path.join(self.model_path, "logs")
        self.log_file_name = f"{self.task_full_name}_node{self.output_depends.node_id}_{self.output_depends.node_ip}.log"
        self.log_file_path = os.path.join(self.log_path, self.log_file_name)
        self.result_path = os.path.join(self.model_path, "result", self.task_full_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.server_args = self._get_launch_server_args()
        self.logger = get_logger(self.log_path, self.log_file_name)
        self.server_available_gpu_mem = None
        
    @classmethod
    def from_task(cls, args:argparse.Namespace, task: Union["TaskOnline","TaskOffline"], local_ip: str):
        return cls(OutputDepends(
            task_id=task.task_id,
            task_name=task.task_name,
            launch_mode=task.launch_mode.value,
            server_cmd=task.server_cmd,
            model_name=task.model_name,
            node_id=0,
            node_ip=f'{local_ip}',
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
                        'speculative-algorithm': [],
                        'speculative-num-steps':[],
                        'speculative-eagle-topk':[],
                        'speculative-num-draft-tokens':[],
                        'Cuda Graph': [],
                        'cuda_graph_max_bs':[],
                        'flash_comm': [],
                        'chunked-prefix-cache':[],
                        'Att Backend': [],
                        'dtype':[],
                        'Parallelism': [],
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
            elif key == 'Torch Compile':
                value.append("ON" if re.search(r"--enable-torch-compile+", command) else "OFF")

            # Speculative decoding
            elif key == 'speculative-algorithm':
                if re.search(r"--speculative-algorithm\s+(\S+)", command):
                    value.append(re.search(r"--speculative-algorithm\s+(\S+)", command).group(1))
                else:
                    value.append(None)
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
            elif key == 'cuda_graph_max_bs':
                RE_match = re.search(r"--cuda-graph-max-bs\s+(\d+)",command)
                if RE_match:
                    value.append(RE_match.group(1))
                else:
                    value.append(None)
            if key == 'chunked-prefix-cache':
                value.append('OFF' if re.search(r"--disable-chunked-prefix-cache+", command) else 'ON')
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
            ret_cmd = f'{ret_cmd} {output_dir_param} "{self.result_path}"'
        if output_file_param != '':
            result_file_jsonl = os.path.join(
                self.result_path, f"{benchmark.get_benchmark_args_str(command)}.jsonl"
            )
            ret_cmd = f'{ret_cmd} {output_file_param} "{result_file_jsonl}"'
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
            df_server_args["available_mem"] = self.server_available_gpu_mem
            df_server_args.pop('tp_size')
            server_args_df = pd.DataFrame(df_server_args)
            metrics = align_metrics_for_dataframe(metrics)
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
