import argparse
from datetime import datetime
import os
from typing import Optional, TYPE_CHECKING, Union
import re
import pandas as pd
import json
import copy

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from PIL import Image
    import_draw_lib_success = True
except ImportError:
    import_draw_lib_success = False


if TYPE_CHECKING:
    from src.task import TaskOnline, TaskOffline

from utils.utils import *

LOGS_SUBPATH = 'logs'
RESULT_SUBPATH = 'result'

TASK_TYPE_UNK = 'tsk_unk'

MODEL_NAME_UNK = 'custom_model'

NOW_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")


class RealProgressManager:
    def __init__(self, args:argparse.Namespace) -> None:
        self.args = args
        self.task = None
        self.task_full_name = ''
        self.fail_reason = 'fail'

        self.real_progress_path = ''
        self.total_real_progress_path = os.path.join(self.args.output_path, f"total_real_progress_file.json")

        self.real_progress_data = ''
        self.total_real_progress_data = {"to_run": "0,0", "docker_tag": self.args.image_tag, "tasks": list()}

        self.online_task_content = {}
        self.offline_task_content = {}
        self.now_time_path = os.path.join(self.args.output_path, NOW_TIME)

    def init(self, task):
        self.task = task
        self.task_full_name = f'{task.task_name.replace("-", "_")}_server{task.task_id}'
        
        t_type = task_type_to_string(self.task.task_type)
        self.online_task_content = {"launch_mode": "online",
                                    "type": t_type,
                                    "simple_param": self.task_full_name,
                                    "server_id": f"{self.task.task_id}",
                                    "cmd": self.task.server_full_cmd,
                                    "client_test": list()}
        self.offline_task_content = {"launch_mode": "offline",
                                     "type": t_type,
                                     "simple_param": self.task_full_name,
                                     "server_id": f"{self.task.task_id}",
                                     "cmd": self.task.server_full_cmd,
                                     "status": "",
                                     "times": 0}
        self.create_real_progress_file()
        self.write_real_progress_start()

    def create_real_progress_file(self) -> None:
        self.real_progress_path = os.path.join(self.args.output_path,
                                               f"{NOW_TIME}",
                                               f"{self.task.model_name}",
                                               OutputManager._task_type_safe(self.task.task_type),
                                               f"real_progress_file.json")

        self.real_progress_data = {"to_run": "0,0", "docker_tag": self.args.image_tag, "tasks": list()}
        if not os.path.exists(self.real_progress_path):
            create_file(self.real_progress_path)

        if not os.path.exists(self.total_real_progress_path):
            create_file(self.total_real_progress_path)

    def write_real_progress_start(self):
        # 因在线和离线的结构不同, 故在此处进行不同类型的处理
        if self.task.launch_mode == TaskLaunchMode.online:
            self.real_progress_data['tasks'].append(copy.deepcopy(self.online_task_content))

            # 此处主要是针对增量情况下的处理
            # 增加判断当前任务是否已被添加
            #   1.已被添加的情况下, 则不添加同样的一条记录
            #   2.未被添加的情况下, 则添加
            is_have_task = False
            # if self.args.specify_task or self.args.incremental_mode:
            #     for task in self.total_real_progress_data['tasks']:
            #         if task["server_id"] == f"{self.task.task_id}":
            #             is_have_task = True
            #             break
            if not is_have_task:
                self.total_real_progress_data['tasks'].append(self.online_task_content)
        elif self.task.launch_mode == TaskLaunchMode.offline:
            self.real_progress_data['tasks'].append(copy.deepcopy(self.offline_task_content))

            # 此处的处理逻辑同在线模式
            # if not self.args.incremental_mode:
            is_same = False
            for cur_task in self.total_real_progress_data['tasks']:
                if cur_task["server_id"] == f"{self.task.task_id}":
                    is_same = True
                    break
            if not is_same:
                self.total_real_progress_data['tasks'].append(self.offline_task_content)

            self.real_progress_data['to_run'] = "%s,%s" % (f"{self.task.task_id+1}", "0")
            self.total_real_progress_data['to_run'] = "%s,%s" % (f"{self.task.task_id+1}", "0")

        self.write_real_progress_config()

    def write_real_progress_bench_serving(self, client_id, is_svr_start=True):
        id = self.task.task_id
        c_id = client_id + 1
        if not is_svr_start:
            id = self.task.task_id + 1
            c_id = client_id
        if (len(self.task.bench_serving) == client_id+1):
            self.real_progress_data['to_run'] = "%s,%s" % (str(self.task.task_id + 1), "0")
        else:
            self.real_progress_data['to_run'] = "%s,%s" % (str(id), str(c_id))

        if (len(self.task.bench_serving) == client_id+1):
            self.total_real_progress_data['to_run'] = "%s,%s" % (str(self.task.task_id + 1), "0")
        else:
            self.total_real_progress_data['to_run'] = "%s,%s" % (str(id), str(c_id))

        if is_svr_start:
            cmd = self.task.bench_serving[client_id]
            self.real_progress_data = self.process_real_progress_bench_serving(self.real_progress_data, cmd)
            self.total_real_progress_data = self.process_real_progress_bench_serving(self.total_real_progress_data, cmd)

        self.write_real_progress_config()

    def process_real_progress_bench_serving(self, content, cmd):
        for cur_task in content['tasks']:
            if cur_task["server_id"] != str(self.task.task_id):
                continue

            is_same = False
            for cur_client in cur_task["client_test"]:
                if cur_client["id"] == str(cmd.get_id()):
                    is_same = True
                    break
            if not is_same:
                data = {"id": str(cmd.get_id()),
                        "cmd": cmd.get_cmd(),
                        "status": "",
                        "times": 0}
                cur_task["client_test"].append(copy.deepcopy(data))
            break
        return content

    def write_real_progress_result(self, result_flag, i):
        if self.task.launch_mode is TaskLaunchMode.online:
            self.real_progress_data = self.process_online_real_progress_result(result_flag, self.real_progress_data, i)
            self.total_real_progress_data = self.process_online_real_progress_result(result_flag, self.total_real_progress_data, i)
        elif self.task.launch_mode is TaskLaunchMode.offline:
            self.real_progress_data = self.process_offline_real_progress_result(result_flag, self.real_progress_data)
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
        if self.real_progress_path is None or self.total_real_progress_path is None:
            return

        with open(self.real_progress_path, 'w') as f:
            json.dump(self.real_progress_data, f, indent=4, ensure_ascii=False)
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
        self.real_progress_data['to_run'] = "%s,%s" % (str(task_id), str(client_id))
        self.total_real_progress_data['to_run'] = "%s,%s" % (str(task_id), str(client_id))
        self.write_real_progress_config()

    def set_fail_reason(self, reason):
        self.fail_reason = reason


class PathManager:
    def __init__(self) -> None:
        self.common_path = ''

        self.node_id = 0
        self.task_full_name = ''

        self.log_file_path = ''
        self.result_common = ''
        self.result_file_path = ''
        self.result_csv_file_path = ''

        self.other_result_file_txt = ''
        self.other_result_file_jsonl = ''
        self.offline_result_file_txt = ''
        self.offline_result_file_jsonl = ''

    # ---------------------------------------------------
    #          设置/获取    公共内容
    # ---------------------------------------------------
    def init(self, output_args, task: Union["TaskOnline", "TaskOffline"]):
        self.common_path = os.path.join(output_args,
                                        f"{NOW_TIME}",
                                        f"{task.model_name}",
                                        OutputManager._task_type_safe(task.task_type))

        self.node_id = [item.is_local for item in task.nodes_used].index(1)
        self.task_full_name = f'{task.task_name.replace("-", "_")}_server{task.task_id}'

        self.log_file_common = os.path.join(self.common_path,
                                          LOGS_SUBPATH)
        self.log_file_name = f"{self.task_full_name}_node{self.node_id}.log"
        self.log_file_path = os.path.join(self.log_file_common, self.log_file_name)
        
        self.result_common = os.path.join(self.common_path, RESULT_SUBPATH)
        self.result_csv_file_path = os.path.join(self.result_common, f"{NOW_TIME}_result.csv")
        acc_path = '' if task.task_type != TaskType.acc else task.acc_type.value
        self.result_file_path = os.path.join(self.result_common,
                                             acc_path,
                                             self.task_full_name)

    def get_task_full_name(self):
        return self.task_full_name

    def get_node_id(self):
        return self.node_id

    # ---------------------------------------------------
    #          获取    result文件路径
    # ---------------------------------------------------
    def get_result_path(self) -> str:
        return self.result_file_path

    def get_result_common_path(self) -> str:
        return self.result_common

    def get_result_csv_file_path(self) -> str:
        return self.result_csv_file_path

    # ---------------------------------------------------
    #       获取 所有情况下txt和jsonl的result文件路径
    # ---------------------------------------------------
    def get_other_result_file_txt_jsonl(self, bench_serving_args_str):
        self.other_result_file_txt = os.path.join(self.get_result_path(), f"{bench_serving_args_str}_result.txt")
        self.other_result_file_jsonl = os.path.join(self.get_result_path(), f"{bench_serving_args_str}_result.jsonl")
        return self.other_result_file_txt, self.other_result_file_jsonl

    def get_offline_result_file_txt_jsonl(self, task: Union["TaskOnline","TaskOffline"]):
        bench_serving_args_str = OutputManager.get_bench_serving_args_str(task.server_cmd)
        self.offline_result_file_txt = os.path.join(self.get_result_path(), f"{bench_serving_args_str}_result.txt")
        self.offline_result_file_jsonl = os.path.join(self.get_result_path(), f"{bench_serving_args_str}_result.jsonl")
        return self.offline_result_file_txt, self.offline_result_file_jsonl

    # ---------------------------------------------------
    #          获取      log文件路径
    # ---------------------------------------------------
    def get_log_path(self) -> str:
        return self.log_file_path

    def get_log_subpath(self) -> str:
        return self.log_file_common


class OutputManager:
    TASK_TYPE_UNK = TASK_TYPE_UNK
    
    @staticmethod
    def get_running_server_args(log='', logger = None):
        if log == '':
            return None,None
        # log = """
        # server_args=ServerArgs(model_path='/mnt/lstore/data/mx/DeepSeek-R1-W8A8/vllm_quant_model/', tokenizer_path='/mnt/lstore/data/mx/DeepSeek-R1-W8A8/vllm_quant_model/', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', trust_remote_code=True, dtype='auto', kv_cache_dtype='auto', quantization=None, quantization_param_path=None, context_length=None, device='cuda', served_model_name='/mnt/lstore/data/mx/DeepSeek-R1-W8A8/vllm_quant_model/', chat_template=None, completion_template=None, is_embedding=False, revision=None, host='127.0.0.1', port=30000, mem_fraction_static=0.86, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=512, max_prefill_tokens=16384, schedule_policy='fcfs', schedule_conservativeness=0.3, cpu_offload_gb=0, page_size=1, tp_size=16, stream_interval=1, stream_output=False, random_seed=419274107, constrained_json_whitespace_pattern=None, watchdog_timeout=300, dist_timeout=None, download_dir=None, base_gpu_id=0, gpu_id_step=1, log_level='info', log_level_http=None, log_requests=False, log_requests_level=0, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, dp_size=4, load_balance_method='round_robin', ep_size=1, dist_init_addr='10.193.78.109:5000', nnodes=2, node_rank=0, json_model_override_args='{}', lora_paths=None, max_loras_per_batch=8, lora_backend='triton', attention_backend='flashinfer', sampling_backend='flashinfer', grammar_backend='xgrammar', speculative_algorithm=None, speculative_draft_model_path=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, disable_radix_cache=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_nccl_nvls=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, disable_mla=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=True, enable_ep_moe=False, enable_deepep_moe=False, deepep_mode=None, enable_torch_compile=False, torch_compile_max_bs=32, cuda_graph_max_bs=80, cuda_graph_bs=None, torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, allow_auto_truncate=False, enable_custom_logit_processor=False, tool_call_parser=None, enable_hierarchical_cache=False, hicache_ratio=2.0, enable_flashinfer_mla=True, enable_flashinfer_grouped_gemm=False, enable_flashmla=False, flashinfer_mla_disable_ragged=False, warmups=None, n_share_experts_fusion=0, disable_shared_experts_fusion=False, debug_tensor_dump_output_folder=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_bootstrap_port=8998)
        # """

        match = re.search(r'server_args=ServerArgs\((.*?)\)', log, re.DOTALL)
        if match:
            args_str = match.group(1)
            raw_args_str = match.group(0)
        else:
            log_msg_level("未找到 server_args=ServerArgs(...) 这一行", logger)
            return None,None


        args_dict = {}
        for param in args_str.split(','):
            param_match = re.match(r'(\w+)=(.*)', param.strip())
            if param_match:
                key = param_match.group(1)
                value = param_match.group(2)
                # if value.startswith("'") and value.endswith("'"):
                #     value = value[1:-1]
                # elif value == 'True':
                #     value = True
                # elif value == 'False':
                #     value = False
                # elif value == 'None':
                #     value = None
                # else:
                #     try:
                #         value = int(value)
                #     except ValueError:
                #         try:
                #             value = float(value)
                #         except ValueError:
                #             pass
                args_dict[key] = value

        # print(f"##get running server args {args_dict}")
        # logger.debug(f"##get running server args {args_dict}")
        return raw_args_str,args_dict

    @staticmethod
    def get_launch_server_args_str(command):
        output_string = ""
        output_string += "-TCON" if re.search(r"--enable-torch-compile+", command) else "-TCOFF"
        output_string += "-NEXTNON" if re.search(r'--speculative-algorithm\s+NEXTN', command) else "-NEXTNOFF"

        splns_match = re.search(r"--speculative-num-steps\s+(\d+)",command)
        if splns_match:
            output_string += f'-splns{splns_match.group(1)}'

        splet_match = re.search(r"--speculative-eagle-topk\s+(\d+)",command)
        if splet_match:
            output_string += f'-splet{splet_match.group(1)}'

        splndt_match = re.search(r"--speculative-num-draft-tokens\s+(\d+)",command)
        if splndt_match:
            output_string += f'-splndt{splndt_match.group(1)}'

        output_string += "-CGOFF" if re.search(r"--disable-cuda-graph+", command) else "-CGON"

        if not re.search(r"--disable-radix-cache+", command):
            output_string += "-HIERAR" if re.search(r"--enable-hierarchical-cache+", command) else "-RADIX"

        if re.search(r"--attention-backend\s+(\S+)", command):
            output_string += "-" + re.search(r"--attention-backend\s+(\S+)", command).group(1)
        if re.search(r"--enable-flashinfer-mla", command):
            output_string += "-flinfmla"

        if re.search(r"--enable-flashmla", command):
            output_string += "-flashmla"

        
        output_string += "-epmoe" if re.search(r"--enable-ep-moe", command) else ""
        output_string += "-dpatt" if re.search(r"--enable-dp-attention", command) else ""


        if re.search(r"--dtype\s+(\S+)", command):
            dtype = re.search(r"--dtype\s+(\S+)", command).group(1)
            output_string += f"-dtype{dtype}"

        CGMB_match = re.search(r"--cuda-graph-max-bs\s+(\d+)",command)
        if CGMB_match:
            output_string += f"-CGMB{CGMB_match.group(1)}" 

        CPS_match = re.search(r"--chunked-prefill-size\s+(\d+)",command)
        if CPS_match:
            output_string += f"-CPS{CPS_match.group(1)}" 


        EBDTP_match = re.search(r"--embedding-tp-size\s+(\d+)",command)
        if EBDTP_match:
            output_string += f"-EBDTP{EBDTP_match.group(1)}"

        output_string += "-flashcomm" if re.search(r"--enable-flash-comm",command) else ""

        MFS_match = re.search(r"--mem-fraction-static\s+(\d+\.\d+|\d+)",command)
        if MFS_match:
            output_string += f"-MFS{MFS_match.group(1)}" 

            
        tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
        ep_size_match = re.search(r"--(ep|ep-size)\s+(\d+)",command)
        dp_size_match = re.search(r"--(dp|dp-size)\s+(\d+)",command)
        if tp_size_match:
            output_string += f"-tp{tp_size_match.group(2)}" 
        if ep_size_match:
            output_string += f"-ep{ep_size_match.group(2)}" 
        if dp_size_match:
            output_string += f"-dp{dp_size_match.group(2)}" 
        
        return output_string

    @staticmethod
    def get_launch_server_args(task_server_id, command):
        server_args = {
                        'Tsid':[str(task_server_id)],
                        'Model':[],
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
            if key == 'Model':
                if re.search(r"--model-path\s+(\S+)", command):
                    model_str = ''
                    if "DeepSeek-R1-BF16_W8A8" in re.search(r"--model-path\s+(\S+)", command).group(1):
                        model_str += 'DS-R1-BF16_W8A8'
                    elif "DeepSeek-R1-BF16" in re.search(r"--model-path\s+(\S+)", command).group(1):
                        model_str += 'DS-R1-BF16'
                    elif "DeepSeek-R1-W8A8" in re.search(r"--model-path\s+(\S+)", command).group(1):
                        model_str += 'DS-R1-W8A8'
                    elif "DeepSeek-R1-awq" in re.search(r"--model-path\s+(\S+)", command).group(1):
                        model_str += 'DS-R1-AWQ'

                value.append(model_str)
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

                tp_str = ''
                dp_str = ''
                ep_str = ''

                assert tp_size_match
                tp_size = tp_size_match.group(2)
                tp_str = f"TP{tp_size}"

                server_args['tp_size'].append(tp_size)

                if dp_size_match:
                    dp_size = dp_size_match.group(2)
                    dp_str = f"DP{dp_size}" 
                    if re.search(r"--enable-dp-attention", command):
                        tp_str = f"TP{int(int(tp_size)/int(dp_size))}" 

                if ep_size_match:
                    ep_str = f"EP{ep_size_match.group(2)}"
                elif re.search(r"--enable-ep-moe", command):
                    ep_str = f"EP{tp_size}"

                value.append(f'{tp_str}{dp_str}{ep_str}')

        # output_string = ""
        # for key, value in server_args.items():
        #     output_string += f'__{key}-{value[0]}'
        # output_string = output_string.replace(" ", "")
        output_string = OutputManager.get_launch_server_args_str(command)
        return server_args, output_string

    @staticmethod
    def _get_bench_serving_args(command):
        input_len_match = re.search(r"--random-input-len\s+(\d+)", command)
        output_len_match = re.search(r"--random-output-len\s+(\d+)", command)
        num_prompt_match = re.search(r"--num-prompts?\s+(\d+)", command)
        rampup_mode_match = re.search(r"rampup-mode\s", command)
        rampup_period_match = re.search(r"--ramp-up-period\s+(\d+\.\d+|\d+)", command)

        input_len = input_len_match.group(1) if input_len_match else "0"
        output_len = output_len_match.group(1) if output_len_match else "0"
        num_prompt = num_prompt_match.group(1) if num_prompt_match else "0"
        rampup_mode = 'ON' if rampup_mode_match else 'OFF'
        rampup_period = rampup_period_match.group(1) if rampup_mode_match else "0"

        return input_len,output_len,num_prompt,rampup_mode,rampup_period

    @staticmethod
    def get_bench_serving_args_str(command):
        input_len,output_len,num_prompt,rampup_mode,rampup_period = OutputManager._get_bench_serving_args(command)
        bench_args_str = f"In{input_len}-out{output_len}-bs{num_prompt}"
        if rampup_mode == 'ON':
            bench_args_str += f'-rp{rampup_period}'
        print(bench_args_str)
        return bench_args_str

    @staticmethod
    def get_other_args(args):
        other_args = {
                        'Build':[args.image_tag],
                    }
        return other_args

    @staticmethod
    def _extract_metrics_from_file(file_path,server_args):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        command_match = re.search(r'^Command:\s*(.+)$', content, re.MULTILINE)
        bench_args = {
                        'batch-size':[],
                        'in-out':[],
                        'rampup_mode':[],
                        'rampup_period':[],
                    }
        if command_match:
            command = command_match.group(1)
        else:
            command = ' '
        input_len,output_len,num_prompt,rampup_mode,rampup_period = OutputManager._get_bench_serving_args(command)
        bench_args['in-out'].append(f'{input_len}-{output_len}')
        bench_args['batch-size'].append(f'{num_prompt}')
        bench_args['rampup_mode'].append(f'{rampup_mode}')
        bench_args['rampup_period'].append(f'{rampup_period}')

        # 这个似乎在外面加过了？
        # running_server_args_str,running_server_args_dict = OutputManager.get_running_server_args(content)
        # if running_server_args_dict:
        #     server_args['mem_frac'][0] = running_server_args_dict['mem_fraction_static']

        metrics = {
            'Infer Backend': re.search(r'Backend:\s+(.+)', content).group(1).strip() if re.search(r'Backend:\s+(.+)', content) else None,
            'Traffic request rate': re.search(r'Traffic request rate:\s+(.+)', content).group(1).strip() if re.search(r'Traffic request rate:\s+(.+)', content) else None,
            'Max request concurrency': re.search(r'Max request concurrency:\s+(.+)', content).group(1).strip() if re.search(r'Max request concurrency:\s+(.+)', content) else None,
            'Successful requests':[int(re.search(r'Successful requests:\s+(\d+)', content).group(1)) if re.search(r'Successful requests:\s+(\d+)', content) else None],
            'Benchmark duration (s)': [float(re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Total input tokens': [int(re.search(r'Total input tokens:\s+(\d+)', content).group(1)) if re.search(r'Total input tokens:\s+(\d+)', content) else None],
            'Total generated tokens': [int(re.search(r'Total generated tokens:\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens:\s+(\d+)', content) else None],
            'Total generated tokens (retokenized)': [int(re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content) else None],
            'Request throughput (req/s)': [float(re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Input token throughput (tok/s)': [float(re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Output token throughput (tok/s)': [float(re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Total token throughput (tok/s)': [float(re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Concurrency': [float(re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content) else None],
            'Accept length': [float(re.search(r'Accept length:\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Accept length:\s+(\d+\.\d+|\d+)', content) else None],
            'Mean E2E Latency (ms)': [float(re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Median E2E Latency (ms)': [float(re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Mean TTFT (ms)': [float(re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Median TTFT (ms)': [float(re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'P99 TTFT (ms)': [float(re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Mean TPOT (ms)': [float(re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Median TPOT (ms)': [float(re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'P99 TPOT (ms)': [float(re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Mean ITL (ms)': [float(re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Median ITL (ms)': [float(re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'P99 ITL (ms)': [float(re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'P95 ITL (ms)': [float(re.search(r'P95 ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P95 ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            'Max ITL (ms)': [float(re.search(r'Max ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Max ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
        }
        tp_size = server_args['tp_size'][0]
        other_indicator = {
            "Decoding TPS(toks/s)": ["{:.2f}".format(1000* metrics['Concurrency'][0] / metrics['Mean ITL (ms)'][0]) if metrics['Mean ITL (ms)'][0] else None],
            "TGS(toks/GPU/s)": ["{:.2f}".format(1000* metrics['Concurrency'][0] / metrics['Mean TPOT (ms)'][0] / int(tp_size))  if metrics['Mean TPOT (ms)'][0]  else None],
            "Interactivity(toks/User/s)": ["{:.2f}".format(1000 / metrics['Mean TPOT (ms)'][0])  if metrics['Mean TPOT (ms)'][0]  else None]
        }

        bench_result_data = {**bench_args, **metrics, **other_indicator}

        return bench_result_data

    @staticmethod
    def _offline_extract_metrics_from_file(file_path,server_args):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        command_match = re.search(r'^Command:\s*(.+)$', content, re.MULTILINE)
        bench_args = {
                        'batch-size':[],
                        'in-out':[],
                        # 'rampup_mode':[],
                        # 'rampup_period':[],
                    }
        if command_match:
            command = command_match.group(1)
        else:
            command = ' '
        input_len,output_len,num_prompt,rampup_mode,rampup_period = OutputManager._get_bench_serving_args(command)
        bench_args['in-out'].append(f'{input_len}-{output_len}')
        bench_args['batch-size'].append(f'{num_prompt}')
        # bench_args['rampup_mode'].append(f'{rampup_mode}')
        # bench_args['rampup_period'].append(f'{rampup_period}')

        # 这个似乎在外面加过了？
        # running_server_args_str,running_server_args_dict = OutputManager.get_running_server_args(content)
        # if running_server_args_dict:
        #     server_args['mem_frac'][0] = running_server_args_dict['mem_fraction_static']

        metrics = {
            'Infer Backend': re.search(r'Backend:\s+(.+)', content).group(1).strip() if re.search(r'Backend:\s+(.+)', content) else None,
            # 'Traffic request rate': re.search(r'Traffic request rate:\s+(.+)', content).group(1).strip() if re.search(r'Traffic request rate:\s+(.+)', content) else None,
            # 'Max request concurrency': re.search(r'Max request concurrency:\s+(.+)', content).group(1).strip() if re.search(r'Max request concurrency:\s+(.+)', content) else None,
            'Successful requests':[int(re.search(r'Successful requests:\s+(\d+)', content).group(1)) if re.search(r'Successful requests:\s+(\d+)', content) else None],
            'Benchmark duration (s)': [float(re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Total input tokens': [int(re.search(r'Total input tokens:\s+(\d+)', content).group(1)) if re.search(r'Total input tokens:\s+(\d+)', content) else None],
            'Total generated tokens': [int(re.search(r'Total generated tokens:\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens:\s+(\d+)', content) else None],
            # 'Total generated tokens (retokenized)': [int(re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content).group(1)) if re.search(r'Total generated tokens \(retokenized\):\s+(\d+)', content) else None],
            'Last generation throughput (tok/s)': [float(re.search(r'Last generation throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Last generation throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Request throughput (req/s)': [float(re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Input token throughput (tok/s)': [float(re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Output token throughput (tok/s)': [float(re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            'Total token throughput (tok/s)': [float(re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Concurrency': [float(re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Concurrency:\s+(\d+\.\d+|\d+)', content) else None],
            # 'Accept length': [float(re.search(r'Accept length:\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Accept length:\s+(\d+\.\d+|\d+)', content) else None],
            # 'Mean E2E Latency (ms)': [float(re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Median E2E Latency (ms)': [float(re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Mean TTFT (ms)': [float(re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Median TTFT (ms)': [float(re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'P99 TTFT (ms)': [float(re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Mean TPOT (ms)': [float(re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Median TPOT (ms)': [float(re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'P99 TPOT (ms)': [float(re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Mean ITL (ms)': [float(re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Median ITL (ms)': [float(re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'P99 ITL (ms)': [float(re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'P95 ITL (ms)': [float(re.search(r'P95 ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'P95 ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
            # 'Max ITL (ms)': [float(re.search(r'Max ITL \(ms\):\s+(\d+\.\d+|\d+)', content).group(1)) if re.search(r'Max ITL \(ms\):\s+(\d+\.\d+|\d+)', content) else None],
        }
        tp_size = server_args['tp_size'][0]
        other_indicator = {
            # "Decoding TPS(toks/s)": ["{:.2f}".format(1000* metrics['Concurrency'][0] / metrics['Mean ITL (ms)'][0]) if metrics['Mean ITL (ms)'][0] else None],
            # "TGS(toks/GPU/s)": ["{:.2f}".format(1000* metrics['Concurrency'][0] / metrics['Mean TPOT (ms)'][0] / int(tp_size))  if metrics['Mean TPOT (ms)'][0]  else None],
            # "Interactivity(toks/User/s)": ["{:.2f}".format(1000 / metrics['Mean TPOT (ms)'][0])  if metrics['Mean TPOT (ms)'][0]  else None]
        }

        bench_result_data = {**bench_args, **metrics, **other_indicator}

        return bench_result_data

    @staticmethod
    def _task_type_safe(task_type:TaskType) -> str:
        if task_type is None: # 可能会忘记设置，
            return TASK_TYPE_UNK
        else:
            return task_type.value

    def __init__(self,args:argparse.Namespace) -> None:
        self.args = args
        self.task = None
        self.task_type = ''
        self.server_args = {}
        self.path_manager = PathManager()
        self.logger = None

    def init_output_file(self) -> None:
        """
        产生一个realprogress/对应task的目录
        应该在start_server/run_slave_launch_server的时候创建
        并完成logger的配置
        """
        self.path_manager.init(self.args.output_path, self.task)
        self.task.set_file_log_subfile(self.path_manager.get_log_subpath())

        self.server_args, _ = OutputManager.get_launch_server_args(self.task.task_id, self.task.server_cmd)
        if not self.server_args['Model'][0] or self.server_args['Model'][0].isspace():
            self.server_args['Model'][0] = self.task.model_name.replace("DeepSeek", "DS")

        self.task_type = OutputManager._task_type_safe(self.task.task_type)

        create_file(self.path_manager.get_log_path())
        # configure_logger(log_file=self.path_manager.get_log_path())
        self.logger = get_logger(self.path_manager.log_file_common, self.path_manager.log_file_name)

    def get_info_for_slave(self) -> dict:
        """
        init_output_file 后可以收集本task的基本信息, 用于slave对齐master log路径
        """
        task_info = {
            'task_id': self.task.task_id,
            'model_name': self.task.model_name,
            'task_type': self.task_type,
            'task_full_name': self.path_manager.get_task_full_name(),
        }
        info = {
            'task_info': task_info,
            'now': NOW_TIME,
            'output_path': self.args.output_path,
        }
        return info

    def extract_result_metrics(self)-> None:
        result_files_path = self.path_manager.get_result_path()
        original_dir = os.getcwd()
        try:
            os.chdir(result_files_path)
            txt_files = [
                entry.name for entry in os.scandir('.')
                if entry.name.endswith('.txt')
            ]
            sorted_files = sorted(txt_files, key=lambda f: os.stat(f).st_mtime)
            result_df = pd.DataFrame()

            server_args = self.server_args

            # with G_MASTER_LOCK:
            oc:OperationContent = self.task.server_cmd_ops[0]
            running_server_args_str,running_server_args_dict = OutputManager.get_running_server_args(log="\n".join(oc.output), logger=self.logger)
            if running_server_args_dict:
                server_args['mem_frac'][0] = running_server_args_dict['mem_fraction_static']
            
            # mmlu 依然要更新 *.txt
            for file_index,txt_file in enumerate(sorted_files):
                if running_server_args_str:
                    with open(txt_file,'a') as f:
                        print(running_server_args_str,file=f)

            for file_index,txt_file in enumerate(sorted_files):
                if self.task.launch_mode is TaskLaunchMode.online: 
                    metrics = OutputManager._extract_metrics_from_file(txt_file,server_args)
                elif self.task.launch_mode is TaskLaunchMode.offline:
                    metrics = OutputManager._offline_extract_metrics_from_file(txt_file,server_args)
                df_server_args = server_args.copy()
                if self.task.launch_mode == TaskLaunchMode.offline:
                    df_server_args["Tsid"][0] = file_index
                df_server_args.pop('tp_size')
                other_args_df = pd.DataFrame(OutputManager.get_other_args(self.args))
                server_args_df = pd.DataFrame(df_server_args)
                bench_result_df = pd.DataFrame(metrics) #might trigger 'used before defined'
                merge_df = pd.concat([other_args_df,server_args_df,bench_result_df],axis=1)
                result_df = pd.concat([result_df,merge_df],ignore_index=True)

            csv_file_name = f'{self.task_type}_result.csv'
            with open(csv_file_name,'w',encoding='utf-8') as csv_file:
                result_df.to_csv(csv_file, index=False)
                self.logger.debug(f"result_csv store in {csv_file_name}")
        finally:
            os.chdir(original_dir)

    def merge_result(self, path_merged, logger):
        result_path = self.path_manager.get_result_common_path()
        if result_path in path_merged:
            return
        path_merged.append(result_path)
        output_data = pd.DataFrame()
        for subdir, dirs, files in os.walk(result_path):
            if subdir == result_path:
                continue
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir,file)
                    data = pd.read_csv(file_path)
                    logger.info(f'merge result {file_path}, len={len(data)}')
                    output_data = pd.concat([output_data, data], ignore_index=True)
        return output_data

    def write_log_file(self):
        """ dump logs from server_cmd_ops[0].output to file """
        outputs = self.task.server_cmd_ops[self.path_manager.get_node_id()].output
        content = "\n".join(outputs) # 暂时无视 是否 store_output
        with open(self.path_manager.get_log_path(), 'a') as f:
            print(content, file=f)

    def set_task(self, task):
        self.task = task


class BenchmarkOutputManager(OutputManager):
    def __init__(self, args: argparse.Namespace, task: Union["TaskOnline","TaskOffline"]) -> None:
        super().__init__(args)
        super().set_task(task)

    def init(self):
        self.init_output_file()
        self.precreate_bench_result_files()

    def precreate_bench_result_files(self) -> None:
        if self.task.launch_mode is TaskLaunchMode.online:
            for command in self.task.bench_serving:
                bench_serving_args_str = OutputManager.get_bench_serving_args_str(command.get_cmd())
                result_file_text, result_file_jsonl = self.path_manager.get_other_result_file_txt_jsonl(bench_serving_args_str)
                create_file(result_file_text)
                create_file(result_file_jsonl)

                with open(result_file_text, "a") as result_file:
                    command_with_result = self.append_result_file_param(command.get_cmd())
                    print(f"Command: {command_with_result}", file=result_file)
        elif self.task.launch_mode is TaskLaunchMode.offline:
            result_file_text, result_file_jsonl = self.path_manager.get_offline_result_file_txt_jsonl(self.task)
            create_file(result_file_text)
            create_file(result_file_jsonl)

            with open(result_file_text, "a") as result_file:
                command_with_result = self.append_result_file_param(self.task.server_cmd)
                print(f"Command: {command_with_result}", file=result_file)

    def append_result_file_param(self, command) -> str:
        if self.task.launch_mode is TaskLaunchMode.offline:
            _, result_file_jsonl = self.path_manager.get_offline_result_file_txt_jsonl(self.task)
            return f"{command} --result-filename {result_file_jsonl}"
        
        bench_serving_args_str = OutputManager.get_bench_serving_args_str(command)
        _, result_file_jsonl = self.path_manager.get_other_result_file_txt_jsonl(bench_serving_args_str)
        return f"{command} --output-file {result_file_jsonl}"

    def write_client_result(self, command, result, is_pass= True) -> None:
        bench_serving_args_str = OutputManager.get_bench_serving_args_str(command)
        result_file_text = os.path.join(self.path_manager.get_result_path(),
                                        f"{bench_serving_args_str}_result.txt")

        command_with_result = self.append_result_file_param(command)

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


class AccOutputManager(OutputManager):
    def __init__(self, args: argparse.Namespace, task: Union["TaskOnline","TaskOffline"]) -> None:
        super().__init__(args)
        super().set_task(task)
        self.acc_ceval_result_file_txt = ''
        self.acc_mmlu_result_file_txt = ''
        self.acc_mmlu_result_file_jsonl = ''

    def init(self):
        self.init_output_file()

        self.acc_ceval_result_file_txt = os.path.join(self.path_manager.get_result_path(), "acc_ceval_result.txt")

        self.acc_mmlu_result_file_txt = os.path.join(self.path_manager.get_result_path(), "acc_mmlu_result.txt")
        self.acc_mmlu_result_file_jsonl = os.path.join(self.path_manager.get_result_path(), "acc_mmlu_result.jsonl")

        self.precreate_bench_result_files()

    def precreate_bench_result_files(self) -> None:
        for command in self.task.bench_serving:
            result_file_text = ''
            if self.task.acc_type == TaskAccType.mmlu:
                result_file_text = self.acc_mmlu_result_file_txt
                create_file(self.acc_mmlu_result_file_txt)
                create_file(self.acc_mmlu_result_file_jsonl)

            elif self.task.acc_type == TaskAccType.ceval:
                result_file_text = self.acc_ceval_result_file_txt
                create_file(result_file_text)

            with open(result_file_text, "a") as result_file:
                command_with_result = self.append_result_file_param(command.get_cmd())
                print(f"Command: {command_with_result}", file=result_file)

    def append_result_file_param(self, command) -> str:
        command = f"{command} --save_dir {self.path_manager.get_result_path()}"
        if self.task.acc_type == TaskAccType.mmlu:
            command = f"{command} --result-file {self.acc_mmlu_result_file_jsonl}"
        return command

    def write_client_result(self, command, result, is_pass=True) -> None:
        result_file_text = ''
        if self.task.acc_type == TaskAccType.mmlu:
            result_file_text = self.acc_mmlu_result_file_txt
        elif self.task.acc_type == TaskAccType.ceval:
            result_file_text = self.acc_ceval_result_file_txt

        command_with_result = self.append_result_file_param(command)

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

    def _get_acc_mmlu_metrics(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        try:
            data = json.loads(content)
            accuracy = data.get("accuracy")
            nsub = data.get("other").get("nsub")
            if accuracy is not None:
                accuracy = float(accuracy)
        except json.JSONDecodeError:
            accuracy = None
            nsub = None
        metrics = {'batch_size':None, 'random_seed':None, 'random_num':None, 'dataset':'mmlu', 'nsub': nsub, 'Accuracy': [accuracy],}
        return metrics

    def _get_acc_ceval_metrics(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        matches = re.findall(r'Accuracy\s*:\s+(\d+\.\d+|\d+)', content)
        last_accuracy = matches[-1] if matches else None
        pattern = r'bs(\d+)_seed(\d+)_num(\d+)'
        match = re.search(pattern, file_path)
        bs = int(match.group(1)) if match else None
        seed = int(match.group(2)) if match else None
        num = int(match.group(3)) if match else None

        metrics = {'batch_size':bs, 'random_seed':seed, 'random_num':num, 'dataset':'ceval', 'nsub': None, 'Accuracy': [float(last_accuracy) if last_accuracy else None]}
        return metrics

    def _extract_acc_metrics_from_file(self, file_path, acc_type: TaskAccType):
        metrics = None
        if acc_type == TaskAccType.mmlu:
            if "acc_mmlu" in file_path:  #?
                metrics = self._get_acc_mmlu_metrics(file_path)
        elif acc_type == TaskAccType.ceval:
            # if "acc_ceval" in file_path:  #?
            # 改为从ceval输出的txt文件里提取
            if 'seed' in file_path and 'num' in file_path:
                metrics = self._get_acc_ceval_metrics(file_path)
        return metrics

    def extract_result_metrics(self)-> None:
        result_files_path = self.path_manager.get_result_path()
        original_dir = os.getcwd()
        try:
            os.chdir(result_files_path)
            txt_files = [
                entry.name for entry in os.scandir('.')
                if entry.name.endswith('result.txt')    #acc的ceval需要排除自带的txt结果
            ]

            sorted_files = sorted(txt_files, key=lambda f: os.stat(f).st_mtime)
            result_df = pd.DataFrame()

            server_args = self.server_args

            # with G_MASTER_LOCK:
            oc:OperationContent = self.task.server_cmd_ops[0]
            running_server_args_str,running_server_args_dict = OutputManager.get_running_server_args(log="\n".join(oc.output), logger=self.logger)
            if running_server_args_dict:
                server_args['mem_frac'][0] = running_server_args_dict['mem_fraction_static']
            
            # mmlu 依然要更新 *.txt
            for file_index,txt_file in enumerate(sorted_files):
                if running_server_args_str:
                    with open(txt_file,'a') as f:
                        print(running_server_args_str,file=f)

            # 目前只有mmlu是从结果的jsonl提取，替换sorted_files进行结果提取
            if self.task.acc_type == TaskAccType.mmlu:
                jsonl_file = [
                    entry.name for entry in os.scandir('.')
                    if entry.name.endswith('.jsonl')
                ]
                sorted_files = sorted(jsonl_file, key=lambda f: os.stat(f).st_mtime)
            elif self.task.acc_type == TaskAccType.ceval:
                txt_file = [
                    entry.name for entry in os.scandir('.')
                    if entry.name.endswith('.txt') and not entry.name.endswith('result.txt')
                ]
                sorted_files = sorted(txt_file, key=lambda f: os.stat(f).st_mtime)

            for file_index,txt_file in enumerate(sorted_files):
                if self.task.launch_mode is TaskLaunchMode.online: 
                    metrics = self._extract_acc_metrics_from_file(txt_file, self.task.acc_type)
                    if metrics is None:
                        continue
                elif self.task.launch_mode is TaskLaunchMode.offline:
                    metrics = OutputManager._offline_extract_metrics_from_file(txt_file,server_args)
                df_server_args = server_args.copy()
                if self.task.launch_mode == TaskLaunchMode.offline:
                    df_server_args["Tsid"][0] = file_index
                df_server_args.pop('tp_size')
                other_args_df = pd.DataFrame(OutputManager.get_other_args(self.args))
                server_args_df = pd.DataFrame(df_server_args)
                bench_result_df = pd.DataFrame(metrics) #might trigger 'used before defined'
                merge_df = pd.concat([other_args_df,server_args_df,bench_result_df],axis=1)
                result_df = pd.concat([result_df,merge_df],ignore_index=True)

            csv_file_name = f'{self.task_type}_result.csv'
            if self.task.acc_type == TaskAccType.mmlu:
                csv_file_name = 'acc_mmlu_result.csv'
            elif self.task.acc_type == TaskAccType.ceval:
                csv_file_name = 'acc_ceval_result.csv'
            with open(csv_file_name,'w',encoding='utf-8') as csv_file:
                result_df.to_csv(csv_file, index=False)
                self.logger.debug(f"result_csv store in {csv_file_name}")
        finally:
            os.chdir(original_dir)


class RampupOutputManager(OutputManager):
    def __init__(self, args: argparse.Namespace, task: Union["TaskOnline","TaskOffline"]) -> None:
        super().__init__(args)
        super().set_task(task)

    def init(self):
        self.init_output_file()
        self.precreate_bench_result_files()

    def precreate_bench_result_files(self) -> None:
        for command in self.task.bench_serving:
            bench_serving_args_str = OutputManager.get_bench_serving_args_str(command.get_cmd())
            result_file_text, result_file_jsonl = self.path_manager.get_other_result_file_txt_jsonl(bench_serving_args_str)
            create_file(result_file_text)
            create_file(result_file_jsonl)

            with open(result_file_text, "a") as result_file:
                command_with_result = self.append_result_file_param(command.get_cmd())
                print(f"Command: {command_with_result}", file=result_file)

    def append_result_file_param(self, command) -> str:
            _, result_file_jsonl = self.path_manager.get_offline_result_file_txt_jsonl(self.task)
            return f"{command} --output-file {result_file_jsonl}"

    def write_client_result(self, command,result, is_pass=True) -> None:
        bench_serving_args_str = OutputManager.get_bench_serving_args_str(command)
        result_file_text = os.path.join(self.path_manager.get_result_path(),
                                        f"{bench_serving_args_str}_result.txt")

        command_with_result = self.append_result_file_param(command)

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


class SearchOutputManager(OutputManager):
    def __init__(self, args: argparse.Namespace, task: Union["TaskOnline","TaskOffline"]) -> None:
        super().__init__(args)
        super().set_task(task)

    def init(self):
        self.init_output_file()
        self.precreate_bench_result_files()

    def precreate_bench_result_files(self) -> None:
        for command in self.task.bench_serving:
            bench_serving_args_str = OutputManager.get_bench_serving_args_str(command.get_cmd())
            result_file_text, result_file_jsonl = self.path_manager.get_other_result_file_txt_jsonl(bench_serving_args_str)
            create_file(result_file_text)
            create_file(result_file_jsonl)

            with open(result_file_text, "a") as result_file:
                command_with_result = self.append_result_file_param(command.get_cmd())
                print(f"Command: {command_with_result}", file=result_file)

    def append_result_file_param(self, command) -> str:
        _, result_file_jsonl = self.path_manager.get_offline_result_file_txt_jsonl(self.task)
        return f"{command} --output-file {result_file_jsonl}"

    def write_client_result(self, command, result, is_pass=True) -> None:
        bench_serving_args_str = OutputManager.get_bench_serving_args_str(command)
        result_file_text = os.path.join(self.path_manager.get_result_path(),
                                        f"{bench_serving_args_str}_result.txt")

        command_with_result = self.append_result_file_param(command)

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

    def parser_single_search_data(self):
        result_files_path = self.path_manager.get_result_path()
        csv_file_name = f'{self.task_type}_result.csv'
        self.logger.debug(f"result_files_path: {result_files_path}")
        file_path = f"{result_files_path}/{csv_file_name}"

        df = pd.read_csv(file_path)
        dp = SearchDataParser()
        dp.max_tpot = self.task.max_tpot
        dp.max_ttft = self.task.max_ttft
        dp.is_filter = True
        file_dir = result_files_path
        optimal_bs = 0
        if dp.is_filter:
            file_dir = f"{result_files_path}/ttft_{dp.max_ttft}_tpot_{dp.max_tpot}"
            self.logger.debug(f"file_dir: {file_dir}")
            if os.path.isdir(file_dir) == False:
                try:
                    os.mkdir(file_dir)
                    self.logger.debug(f"folder '{file_dir}' creation successful")
                except FileExistsError:
                    self.logger.debug(f"folder '{file_dir}' already exist")
                except OSError as e:
                    self.logger.debug(f"creation failed:{e}") 
            result = dp.parser_ttft_tpot_data(df,file_dir)
            optimal_bs = result[1]
            if import_draw_lib_success:
                dp.plot_specified_data(result[0],file_dir)
        dp.is_filter = False
        dp.optimal_bs = optimal_bs
        if import_draw_lib_success:
            dp.plot_specified_data(df,result_files_path)

    def merge_online_search_result(self, task_type: TaskType, max_ttft, max_tpot) -> None:
        try:
            result_path = self.path_manager.get_result_common_path()
            all_data = pd.DataFrame()
            for subdir, dirs, files in os.walk(result_path):
                if subdir == result_path:
                    continue
                # dir_name = os.path.basename(subdir)
                for file in files:
                    if file.endswith('.csv') and "ttft" not in file and "tpot" not in file:
                        file_path = os.path.join(subdir,file)
                        data = pd.read_csv(file_path)
                        all_data = pd.concat([all_data, data], ignore_index=True)

            output_path = self.path_manager.get_result_csv_file_path()
            with open(output_path,'w',encoding='utf-8') as csv_file:
                all_data.to_csv(csv_file, index=False)
            self.logger.debug(f"{OutputManager._task_type_safe(task_type)} result_csv store in {output_path}")
            self.parser_total_search_data(result_path, max_ttft, max_tpot)
                
        except Exception as e:
            self.logger.info(f"merge_search_result exception {e}")

    def parser_total_search_data(self, result_files_path, max_ttft, max_tpot):
        self.logger.debug(f"result_files_path: {result_files_path}")
        file_path = self.path_manager.get_result_csv_file_path()

        df = pd.read_csv(file_path)
        dp = SearchDataParser()
        dp.max_tpot = max_tpot
        dp.max_ttft = max_ttft
        dp.is_filter = True
        file_dir = result_files_path
        optimal_bs = 0
        if dp.is_filter:
            file_dir = f"{result_files_path}/total_ttft_{dp.max_ttft}_tpot_{dp.max_tpot}"
            self.logger.debug(f"file_dir: {file_dir}")
            if os.path.isdir(file_dir) == False:
                try:
                    os.mkdir(file_dir)
                    self.logger.debug(f"folder '{file_dir}' creation successful")
                except FileExistsError:
                    self.logger.debug(f"folder '{file_dir}' already exist")
                except OSError as e:
                    self.logger.debug(f"creation failed:{e}") 
            result = dp.parser_ttft_tpot_data(df,file_dir)
            optimal_bs = result[1]
            if import_draw_lib_success:
                dp.plot_specified_data(result[0],file_dir)
        dp.is_filter = False
        dp.optimal_bs = optimal_bs
        if import_draw_lib_success:
            dp.plot_specified_data(df,result_files_path)


class SearchDataParser():

    def __init__(self) -> None:
        self.max_ttft = None
        self.max_tpot = None
        self.is_filter = False
        self.now = datetime.now().strftime("%Y%m%d_%H%M")
        self.optimal_bs = None #最优batch size
 
    def parser_ttft_tpot_data(self,df, target_file_path="."):
        if self.is_filter == False:
            return (df, None)
        self.logger.debug("start parser ttft tpot data")
        if df is None or df.empty:
            self.logger.debug("parser ttft tpot data: df is null or empty")
            return (None, None)
        if not target_file_path.strip():
            self.logger.debug("target file path is null !!!")
            return (None, None)
        optimal_bs = 0
        try:
            target_data = df[(df["Mean TTFT (ms)"] < float(self.max_ttft)) & (df["Mean TPOT (ms)"] < float(self.max_tpot))]
            optimal_data = target_data[target_data["batch-size"] == target_data["batch-size"].max()]
            optimal_bs = optimal_data["batch-size"].max()
            output_file = f"{target_file_path}/ttft_{self.max_ttft}_tpot_{self.max_tpot}_result.csv"
            self.logger.debug(f"output_file:{output_file}")
            with open(output_file,'w+',encoding='utf-8') as csv_file:
                target_data.to_csv(csv_file, index=False, encoding="utf-8")

            optimal_data_file = f"{target_file_path}/max_ttft_{self.max_ttft}_max_tpot_{self.max_tpot}_result.csv"
            optimal_data.to_csv(optimal_data_file, index=False, encoding="utf-8")

        except Exception as e:
            self.logger.debug(f"parser ttft tpot data: exception {e}")
            return (None, None)
        self.logger.debug(f"ttft_tpot_csv store in {output_file}")
        return (target_data, optimal_bs)
  
    def plot_specified_data(self,df,file_path="."):
        if df is None or df.empty:
            self.logger.debug("plot specified data: df is null or empty")
        else:
            bs = df["batch-size"].values
            ttft_key = "Mean TTFT (ms)"
            tpot_key = "Mean TPOT (ms)"
            tus_key = "Interactivity(toks/User/s)"
            tgs_key = "TGS(toks/GPU/s)"
            ott_key = "Output token throughput (tok/s)"
            img_dir = f"{file_path}/{self.now}_"
            ttft_tpot_img_path = f"{file_path}/{self.now}_ttft_tpot.png"
            if self.max_ttft and self.max_tpot and self.is_filter:
                 img_dir = f"{file_path}/{self.now}_ttft_{self.max_ttft}_tpot_{self.max_tpot}_"
                 ttft_tpot_img_path = f"{file_path}/{self.now}_ttft_{self.max_ttft}_tpot_{self.max_tpot}_ttft_tpot.png"

            self.draw_image(bs,df[ttft_key].values,"Time to First Token",ttft_key,f"{img_dir}ttft.png",True,False)
            self.draw_image(bs,df[tpot_key].values,"Time per Output Token (excl. 1st token)",tpot_key,f"{img_dir}tpot.png",False, True)
            self.draw_image(bs,df[ott_key].values,ott_key,ott_key,f"{img_dir}ott.png")
            self.draw_image(bs,df[tgs_key].values,tgs_key,tgs_key,f"{img_dir}tgs.png")
            self.draw_image(bs,df[tus_key].values,tus_key,tus_key,f"{img_dir}tus.png")
        
            img_list = [f"{img_dir}ttft.png",f"{img_dir}tpot.png"]
            self.stitch_images_horizontally(img_list,ttft_tpot_img_path)

            

    def draw_image(self,x, y, title="", legend_label="", image_path="",is_ttft=False, is_tpot=False):
        self.logger.debug(f"start draw image {title}")
        if not image_path.strip():
            self.logger.debug("image path is null !!!")
        else:
            plt.cla()         # 清除当前坐标轴
            plt.clf()         # 清除当前图形
            plt.close('all')  # 关闭所有图形窗口
            mpl.rcParams.update(mpl.rcParamsDefault)  # 恢复默认配置
            plt.style.use('default')  # 使用默认样式 
            plt.plot(x, y, marker='o', linestyle='--', color = "red",label=legend_label)
           
            # 设置每个点的值
            for xt, yt in zip(x, y):
                plt.text(xt,yt,f"{yt}",fontsize = 8)

            if self.optimal_bs and self.optimal_bs > 0:
                if is_ttft:
                    index = x.tolist().index(self.optimal_bs)
                    if index > -1:
                        plt.plot(x[index], y[index], 'go', markersize=8)# 'go'表示绿色圆形标记
                elif is_tpot:
                    index = x.tolist().index(self.optimal_bs)
                    if index > -1:
                        plt.plot(x[index], y[index], 'go', markersize=8) # 'go'表示绿色圆形标记
      
            # 添加标题和坐标轴标签
            plt.title(title)
            plt.xlabel('batch_size')

            # 设置x轴只显示指定的刻度值 
            plt.xticks(x)
            # 添加网格
            plt.grid(True)

            # 添加图例
            plt.legend()
            plt.savefig(image_path)

    @staticmethod     # 拼接图片（默认垂直）
    def stitch_images_horizontally(image_paths, output_path, is_vertically=True):
        
        # 打开所有图片
        images = [Image.open(path) for path in image_paths]
        
        # 获取所有图片的宽度和高度
        widths, heights = zip(*(img.size for img in images))
        
        # 计算拼接后图片的总宽度和最大高度
        if is_vertically:
            total_width = max(widths)
            max_height = sum(heights)
        else:
            total_width = sum(widths)
            max_height = max(heights)
        
        # 创建空白画布
        new_image = Image.new('RGB', (total_width, max_height))
        
        # 拼接图片
        x_offset = 0
        y_offset = 0
        for img in images:
            if is_vertically:
                x_offset = (total_width - img.width) // 2
                new_image.paste(img, (x_offset, y_offset))
                y_offset += img.height
            else:
                new_image.paste(img, (x_offset, 0))
                x_offset += img.width
        
        # 保存结果
        new_image.save(output_path)
        for img_pt in image_paths:
            if os.path.exists(img_pt):
                try:
                    os.remove(img_pt)
                    print("remove image done")
                except OSError as e:
                    print(e)
