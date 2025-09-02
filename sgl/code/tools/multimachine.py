import argparse
import os
import subprocess
from typing import List, Dict, Tuple
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import socket
import signal
import time
import pandas as pd

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的文件夹路径
current_dir = os.path.dirname(current_file_path)
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) # 时间戳

class Machine:

    def __init__(self, config, ip, gpu_count, unused_gpu_count):
        self.config = config
        self.ip = ip
        self.is_local = False
        self.gpu_count = gpu_count
        self.unused_gpu_count = unused_gpu_count
        self.useed_gpu_index = -1
    
    def reset(self):
        self.unused_gpu_count = self.gpu_count
        self.useed_gpu_index = -1
class Task:
    
    def __init__(self,data, path, gpu_count, index):
        self.data = data
        self.path = path
        self.gpu_count = gpu_count
        self.index = index

class Multimachine:
    def __init__(self, args:argparse.Namespace):
        self.args:argparse.Namespace = args
        self.total_config = []
        self.machine_list: list[Machine] = []
        self.total_gpu_count = 0
        self.expected_total_gpu_count = 0
        self.master_nodes = []
        self.command_list = []
        self.cancel_cmds = []
        

    def run(self) -> None:
        # 注册信号处理函数，监听SIGINT信号（对应Ctrl+C）
        signal.signal(signal.SIGINT, self.handle_ctrl_c)
        local_ip = get_ip_from_command()
        if not local_ip:
            print("Error:## unable to get local ip !")
            exit(1)
        print(f"local ip: {local_ip}")
        total_task_list = list()
        # 读取测试任务信息
        tasks_path_list = self.args.tasks_config
        for path in tasks_path_list:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.total_config.append({"data":data,"path":path})
                tasks = data["tasks"]
                for i,task in enumerate(tasks):
                    gpu_count = self.get_expected_gpu_count(task)
                    total_task_list.append(Task(task,path,gpu_count,i))
                    self.expected_total_gpu_count += gpu_count
      
        # 读取节点信息   
        with open(self.args.machine_config,"r", encoding="utf-8") as f:
            data = json.load(f)
            for info in data["machine_info"]:
                ip = info["ip"]
                username = self.args.user  # 远程用户名
                count = self.get_gpu_count(ip, username, ip in local_ip)
                machine = Machine(config=info, ip=ip, gpu_count=count,unused_gpu_count=count)
                machine.is_local =  ip in local_ip
                self.machine_list.append(machine)
                self.total_gpu_count += count
        if not total_task_list or not self.machine_list:
            print("Task config or machine list config is null !!!")
            return
        print(f"Total of all nodes:{self.total_gpu_count} GPU")
        print(f"Total required for all tasks:{self.expected_total_gpu_count} GPU")
        print([task.gpu_count for task in total_task_list])
        new_json_dir = f"{os.path.dirname(tasks_path_list[0])}/section"
        delete_folder(new_json_dir)
        create_folder(new_json_dir)
        rounds, schedule = self.calculate_task_rounds(self.total_gpu_count, total_task_list)
        print(f"总可用GPU卡数: {self.total_gpu_count}")
        print(f"任务所需卡数: {[task.gpu_count for task in total_task_list]}")
        print(f"总轮数: {rounds}")
        print("每轮任务分配:")
        server_port = 0
        sgl_port = 3000
        for i, round_tasks in enumerate(schedule, 1):
            for machine in self.machine_list:
                machine.reset()
            print(f"  第{i}轮: 任务卡数 {[task.gpu_count for task in round_tasks]}, 总使用 {sum([task.gpu_count for task in round_tasks])} 卡")
            run_task, new_server_prot, new_sgl_prot = self.split_config(round_tasks,round_tasks[0].path,server_port,sgl_port)
            print("\n")
            server_port = new_server_prot
            sgl_port = new_sgl_prot
            self.run_docker(local_ip,run_task)
        self.merge_result_data()
        
    def check_file(self,ip="",username="root",path=""):
        target_path = path
        if check_file_or_dir_is_exist(ip,username,path=target_path) == False:
            create_folder(os.path.dirname(target_path))
            print(target_path, path)
            reslut = copy_file_or_dir_to_remote(ip,username,source_path=path,target_path=target_path)
            print(f"copy config: {reslut}")
        else:
            print(f"file \"{target_path}\" already exist")
            
    def calculate_task_rounds(self, total_gpus, total_tasks):
        """
        计算任务需要的运行轮次，并分配每轮运行的任务
        
        参数:
            total_gpus: 可用的总GPU卡数
            task_gpus: 每个任务需要的GPU卡数列表
            
        返回:
            rounds: 总轮数
            schedule: 每轮运行的任务列表，每个元素是该轮运行的任务所需卡数列表
        """
        # 复制并按从大到小排序任务
        sorted_tasks = sorted(total_tasks, key=lambda s:s.gpu_count,reverse=True)
        remaining_tasks = sorted_tasks.copy()
        schedule = []
        
        # 循环分配任务直到所有任务都被分配
        while remaining_tasks:
            current_round = []
            current_used = 0
            
            # 尝试为当前轮次分配任务
            i = 0
            while i < len(remaining_tasks):
                task = remaining_tasks[i]
                # 检查是否可以加入当前轮次
                if current_used + task.gpu_count <= total_gpus:
                    current_round.append(task)
                    current_used += task.gpu_count
                    remaining_tasks.pop(i)  # 从剩余任务中移除
                else:
                    i += 1  # 不能加入，检查下一个任务
            
            if current_round:  # 确保当前轮次有任务
                schedule.append(current_round)
            else:
                # 如果没有任务能被分配，说明有任务所需卡数超过总卡数
                raise ValueError(f"任务所需卡数 {remaining_tasks[0]} 超过可用总卡数 {total_gpus}")
        
        return len(schedule), schedule
    
    def run_docker(self,local_ip,run_task):
        all_run_cmds = []
        all_docker_rm_cmds = []
        port = 0
        user = self.args.user      
        for i, task in enumerate(run_task):
            tasks_config = task["tasks_config_path"]
            machine_config = task["machine_config_path"]
            file_name_with_ext = os.path.basename(tasks_config)
            file_name, _ = os.path.splitext(file_name_with_ext)
            container_name = f"{self.args.container_name}_{file_name}_{i}"
            ip_list = task["ip"]
            if not ip_list:
                print("Error: Please configure the machine information !!!")
                return
            if port == 0:
                port = self.args.port
            master_node_ip = ip_list[0]
            is_local_machine = master_node_ip in local_ip
            output_path = f"{self.args.output_path}/{timestamp}_multi"
            node = {"ip":master_node_ip,"is_local":is_local_machine,"path":output_path,"task_type":file_name}
            if is_local_machine:
                if node not in self.master_nodes:
                    self.master_nodes.insert(0,node) 
            else:
                if node not in self.master_nodes:
                    self.master_nodes.append(node)
            cmd = self.create_cmd(container_name, tasks_config, machine_config, port, output_path)
            if is_local_machine:
                create_folder(output_path)
            else:
                create_remote_folder(master_node_ip,user,output_path)        
                cmd =  f'ssh -o StrictHostKeyChecking=no {user}@{master_node_ip} "cd {current_dir} ; {cmd}"'

            port +=10000
            docker_rm_cmd = f"docker rm {container_name}"
            docker_stop_cmd = f"docker stop {container_name}"
            for ip in ip_list:
                if ip not in local_ip:
                    docker_rm_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "{docker_rm_cmd}"'
                    docker_stop_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "{docker_stop_cmd}"'
                    self.check_file(ip,user,tasks_config)
                    self.check_file(ip,user,machine_config)
                    if check_file_or_dir_is_exist(ip,user,output_path) == False:
                        create_remote_folder(ip,user,output_path)            
                self.cancel_cmds.append(docker_stop_cmd)
                self.cancel_cmds.append(docker_rm_cmd)
                all_docker_rm_cmds.append(docker_rm_cmd)
                
            all_run_cmds.append(cmd)   
           
        print("Start concurrent execution of commands...")
        
        task_run_results = run_commands_concurrently(
            commands=all_run_cmds,
            max_workers=len(all_run_cmds)  # 限制最大并发数
        )

        # 输出最终执行结果
        print("\n===== The test task command has been executed successfully =====")
        for cmd, success in task_run_results.items():
            status = "succeed" if success else "fail"
            print(f"command: {cmd} -> {status}")
   
        docker_rm_results = run_commands_concurrently(
            commands=all_docker_rm_cmds,
            max_workers=len(all_docker_rm_cmds)  # 限制最大并发数
        )
        
        print("\n===== All commands have been executed =====")
        for cmd, success in docker_rm_results.items():
            status = "succeed" if success else "fail"
            print(f"command: {cmd} -> {status}")
         
    def merge_result_data(self):
        time_str = timestamp
        print(time_str)
        dst_path = ""
        source_path = ""
        task_types = set()
    
        for node in self.master_nodes:
            is_local = node["is_local"]
            task_type = node["task_type"]
            if task_type == "rampup_bench":
                task_type = "rampup"
            task_types.add(task_type)
            ip = node["ip"]
            path = node["path"]
            if not dst_path:
                dst_path = f"{self.args.output_path}/{time_str}"
                if is_local:
                    create_folder(dst_path)
                else:
                    create_remote_folder(ip,self.args.user,dst_path)
            
            copy_file_or_dir_to_local(ip, self.args.user,path,dst_path)
            source_path = f"{dst_path}/{time_str}_multi"
            path_result = get_specified_subfolders_or_file_recursive_os(source_path, task_type)
            if not path_result:
                continue
            for p in path_result:
                # rsync -av --progress 合并文件夹，同名文件默认保留最新
                cmd = f"rsync -a {p} {dst_path}" 
                run_command_realtime(cmd,thread_name="rsync")
            if is_local:
                delete_folder(path)
            else:
                run_command_realtime(f"ssh {self.args.user}@{ip} \" rm -rf {path}\"",thread_name="rm")
        
        total_file = None
        total_file_name = "total_real_progress_file.json"
        with open(f"{source_path}/{total_file_name}","r", encoding="utf-8") as f:
            total_file = json.load(f)
        total_tasks = []
        for tt in task_types:
            target_name = f"{tt}_result.csv"
            if tt == "acc":
                acc_sub = ["ceval","mmlu"]
                acc_all_result = []
                for sub in acc_sub:
                    target_name = f"{tt}_{sub}_result.csv"
                    file_path_result = find_files_by_name(f"{dst_path}/{tt}",target_name)
                    acc_all_result.extend(file_path_result)
                if acc_all_result:
                    first = os.path.dirname(acc_all_result[0])
                    output_path = os.path.dirname(os.path.dirname(first))
                    for f in find_files_by_name(output_path,"_result.csv",False):
                        os.remove(f)
                    merge_file_path = f"{output_path}/{time_str}_result.csv"
                    merge_csv_files(acc_all_result, merge_file_path,test_type=tt)
            else:
                file_path_result = find_files_by_name(f"{dst_path}/{tt}",target_name)
                if file_path_result:
                    first = os.path.dirname(file_path_result[0])
                    output_path = os.path.dirname(first)
                    for f in find_files_by_name(output_path,"_result.csv",False):
                        os.remove(f)
                    merge_file_path = f"{output_path}/{time_str}_result.csv"
                    merge_csv_files(file_path_result, merge_file_path,test_type=tt)
            if tt == "search":
                folders = get_specified_subfolders_or_file_recursive_os(f"{dst_path}/{tt}","total_ttft",is_fuzzy_match=True)
                if folders:
                    ttft = ""
                    tpot = ""
                    total_ttft_tpot_path = folders[-1]
                    subs = total_ttft_tpot_path.split("/")[-1].split("_")
                    if len(subs) == 5:
                        ttft = subs[2]
                        tpot = subs[-1]
                        try:
                            df = pd.read_csv(merge_file_path, encoding="utf-8")
                            target_data = df[(df["Mean TTFT (ms)"] < float(ttft)) & (df["Mean TPOT (ms)"] < float(tpot))]
                            optimal_data = target_data[target_data["batch-size"] == target_data["batch-size"].max()]
                            output_file = f"{total_ttft_tpot_path}/ttft_{ttft}_tpot_{tpot}_result.csv"
                            with open(output_file,'w+',encoding='utf-8') as csv_file:
                                target_data.to_csv(csv_file, index=False, encoding="utf-8")

                            optimal_data_file = f"{total_ttft_tpot_path}/max_ttft_{ttft}_max_tpot_{tpot}_result.csv"
                            optimal_data.to_csv(optimal_data_file, index=False, encoding="utf-8")
                            print(output_file, optimal_data_file)
                        except Exception as e:
                            print(f"csv file fail: {str(e)}")
                                       
            real_progress_file_name = f"{tt}/real_progress_file.json"
            files = get_specified_subfolders_or_file_recursive_os(source_path, real_progress_file_name,True)
            files = sorted(files)
            real_progress_file = None
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    if not real_progress_file:
                        real_progress_file = json.load(f)
                    else:
                        tasks = real_progress_file["tasks"]
                        data = json.load(f)
                        tasks.extend(data["tasks"])
            if real_progress_file:
                with open(f"{dst_path}/{real_progress_file_name}", 'w', encoding='utf-8') as f:
                        json.dump(
                            real_progress_file,
                            f,
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=False  # 不排序键，保持原字典顺序
                        )
            total_tasks.extend(real_progress_file["tasks"])
            
        total_file["tasks"] = total_tasks
        if total_file:
            with open(f"{dst_path}/{total_file_name}", 'w', encoding='utf-8') as f:
                        json.dump(
                            total_file,
                            f,
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=False  # 不排序键，保持原字典顺序
                        )
        delete_folder(source_path)      
       
    def split_config(self,tasks,path,server_port,sgl_port):
        """
        切分任务
        """
        a = "=="
        run_task = []
        for task in tasks:
            value = task.data
            file_name_with_ext = os.path.basename(task.path)
            file_name, ext = os.path.splitext(file_name_with_ext)
            
            if server_port == 0:
                server_port = int(value["server_port"])
            machine_info = []
            use_machine = None
            expected_gpu_count = task.gpu_count
            print(f"{a*35} START {a*35}")
            print(f"The number of cards required for a single task: {expected_gpu_count}")
            
            count_expected = 0
            for machine in self.machine_list:
                if machine.unused_gpu_count == 0:
                    continue
                print(f"Machine:{machine.ip} number of idle cards: {machine.unused_gpu_count}")
                if machine.unused_gpu_count >= expected_gpu_count:
                    use_machine = machine
                    machine_info.append(machine.config)
                    print(f"Number of allocated cards: {expected_gpu_count}")
                    break
                else:
                    count_expected += machine.gpu_count
                    machine.unused_gpu_count = 0
                    if machine.is_local:
                        machine_info.insert(0,machine.config)
                    else:
                        machine_info.append(machine.config)
                    if count_expected >= expected_gpu_count:
                        print(f"Number of allocated cards: {expected_gpu_count}")
                        break
                    
            new_json_dir = f"{os.path.dirname(path)}/section/{file_name}_{task.index}"
            create_folder(new_json_dir)
            new_json_path = f"{new_json_dir}/{file_name}{ext}"
            cvd = ""
            if use_machine != None:
                gpu_index = use_machine.useed_gpu_index+1
                devices_index = ",".join(map(str, range(gpu_index, gpu_index + expected_gpu_count)))
                print(f"gpu_index:{gpu_index} expected_gpu_count:{expected_gpu_count} specified gpu index: {devices_index}")
                cvd = f"CUDA_VISIBLE_DEVICES={devices_index} "
                use_machine.unused_gpu_count -= expected_gpu_count
                use_machine.useed_gpu_index += expected_gpu_count
                print(f"Machine:{machine.ip} use the graphics card index: {use_machine.useed_gpu_index}")
                print(f"Machine:{machine.ip} number of idle cards: {use_machine.unused_gpu_count}")
            
            value["server_port"] = f"{server_port}"
            if value["launch_mode"] == "offline":
                server_base = value["server_base"]
                cmb = server_base["command_base"]
                server_base["command_base"] = f"{cvd}{cmb} --port {sgl_port}"
                print(server_base["command_base"])
            else:
                launch_server = value["launch_server"]
                command_base_list = launch_server["command_base"]
                new_cmd_list = []
                for cmb in command_base_list:
                    cmb = f"{cvd}{cmb}  --port {sgl_port}"
                    new_cmd_list.append(cmb)
                launch_server["command_base"] = new_cmd_list

                benchmark = value["benchmark"]
                command_base = benchmark["command_base"]
                benchmark["command_base"] = f"{cvd}{command_base} --port {sgl_port}"
                print(launch_server["command_base"])
                print(benchmark["command_base"])
            
            print(f'server_port: {value["server_port"]}')
            
            # 写入JSON文件
            new_tasks_json = {"tasks":[value]}
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    new_tasks_json,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=False  # 不排序键，保持原字典顺序
                )
            machine_config_path = f"{new_json_dir}/machines.json"
            machines_json = {"machine_info":machine_info}
            with open(machine_config_path, 'w', encoding='utf-8') as f:
                json.dump(
                    machines_json,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=False  # 不排序键，保持原字典顺序
                )
            if machine_info:
                run_task.append({"tasks_config_path":new_json_path, "machine_config_path":machine_config_path,"ip":[machine["ip"] for machine in machine_info]})
                sgl_port +=10
                server_port += 1000
            print(f"{a*35} END {a*35}")   
        return run_task, server_port, sgl_port

    def create_cmd(self, container_name, tasks_config, machine_config, port, output_path="") ->str:
        incremental = "--incremental-mode" if self.args.incremental_mode else ""
        specify = "--specify-task" if self.args.specify_task else ""
        rm_exist_docker = "--rm-exist-docker" if self.args.rm_exist_docker else ""
        pull_images = "--pull-images" if self.args.pull_images else ""
        prepare_docker_cmds = " ".join(f'"{item}"' for item in self.args.prepare_docker_cmds)
        cmd = f'python3 -m start_docker\
                --container-name {container_name}\
                --container-images {" ".join(self.args.container_images)}\
                --container-cycles {self.args.container_cycles}\
                --docker-v {" ".join(self.args.docker_v)}\
                --prepare-docker-cmds {prepare_docker_cmds}\
                {pull_images}\
                {rm_exist_docker}\
                --target-path {self.args.target_path}\
                --output-path {output_path}\
                --user {self.args.user}\
                {incremental}\
                {specify}\
                --tasks-config {tasks_config}\
                --machine-config {machine_config}\
                --port {port}\
                '
        return cmd
    
    def get_expected_gpu_count(self,task) ->int:
        world_size = task["world_size"]
        if not world_size:
            if task["launch_mode"] == "offline":
                command = "".join(task["server_base"]["param"])
                tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
                return int(tp_size_match.group(2))
            else:
                command = "".join(task["launch_server"]["enable_parallel"])
                tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
                return  int(tp_size_match.group(2))
        else:
            return int(world_size)
        
    def get_gpu_count(self,ip, username, port=22, is_local=True): 
        #  "mx-smi --show-pcie" 
        gpu_count = 0
        cmd = "mx-smi"
        if not is_local:
            cmd = f'ssh -o StrictHostKeyChecking=no {username}@{ip} "mx-smi"'
        # print(cmd)
        try:
            pp = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            info = pp.stdout.read()
            # gpu_count = info.count("GPU#")
            for line in info.split('\n'):
                # 查找包含"Attached GPUs"的行
                if "Attached GPUs" in line:
                    # 分割行，获取冒号后面的部分并去除空格
                    gpu_count = int(line.split(':')[-1].strip())
                    break
            return gpu_count
            
        except FileNotFoundError:
            print(f"Error: Command '{cmd}' It does not exist or the path is incorrect")
        except PermissionError:
            print(f"Error: No permission to execute commands '{cmd}'")
        except OSError as e:
            print(f"System error:{e.strerror}(error code:{e.errno})")
        except Exception as e:
            print(f"Unexpected error:{str(e)}")

        return gpu_count

    def handle_ctrl_c(self, signal_num, frame):
        """处理Ctrl+C事件的回调函数"""
        print("\n收到Ctrl+C终止信号，正在安全退出...")
        # 在这里添加清理操作，如关闭文件、释放资源等
        
        for cmd in self.cancel_cmds:
            success, _ = run_command_realtime(cmd,"cancel")
            status = "succeed" if success else "fail"
            print(f"{cmd} execute {status}")
        exit(0)
        
def run_command_realtime(command: str, thread_name: str = None) -> Tuple[bool, str]:
    """
    实时输出单个命令的执行过程
    
    参数:
        command: 要执行的命令
        thread_name: 线程名称（用于区分输出）
        
    返回:
        (是否成功, 命令字符串)
    """
    thread_name = thread_name or f"thread-{threading.current_thread().ident}"
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 实时读取输出，添加线程标识
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                # 打印时添加线程名称，区分不同命令的输出
                print(f"[{thread_name}] {line.strip()}")
        
        success = process.returncode == 0
        if not success:
            print(f"[{thread_name}] Command execution failed. Return code: {process.returncode}")
        return (success, command)
    
    except Exception as e:
        print(f"[{thread_name}] execute exception: {str(e)}")
        return (False, command)

def run_commands_concurrently(
    commands: List[str],
    max_workers: int = None
) -> Dict[str, bool]:
    """
    并发执行多个命令，保持实时输出
    
    参数:
        commands: 命令列表
        max_workers: 最大并发数（默认根据CPU核心数确定）
        
    返回:
        字典，键为命令，值为执行是否成功
    """
    results = {}
    futures = {}
    # 线程池执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    
        for i, cmd in enumerate(commands):
            # 提交任务到线程池
            future = executor.submit(
                run_command_realtime, 
                cmd, 
                thread_name=f"cmd-{i+1}"  # 为每个命令指定线程名
            )
            futures[future] = cmd
            # 不是最后一个任务时，添加提交间隔
            if i < len(commands) - 1:
                time.sleep(1) # 设置任务提交间隔时间(秒)
                
        # 等待任务完成并收集结果
        for future in as_completed(futures):
            cmd = futures[future]
            try:
                success, _ = future.result()
                results[cmd] = success
            except Exception as e:
                print(f"command '{cmd}' Result acquisition failed: {str(e)}")
                results[cmd] = False
    
    return results

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Multi-machine startup")
    # 容器相关参数
    parser.add_argument("--container-name", type=str, default="", help="container name")
    parser.add_argument("--container-images", type=str,nargs='*', default=[], help="eg:--docker-config  repository1:tag    repository2:tag   repository1:tag   repository2:tag")
    parser.add_argument("--container-cycles", type=int, default=1, help="Number of cycles")
    parser.add_argument("--docker-v", type=str,nargs='*', default=[], help="Mounting path")
    parser.add_argument("--rm-exist-docker", action="store_true", help="force remove exist docker")
    parser.add_argument("--prepare-docker-cmds",nargs='*',type=str,default=[],help="Command for initializing the environment")
    parser.add_argument("--target-path", type=str, default="/workspace/ModelZoo.LLM.Inference/", help="Target work path")
    parser.add_argument("--pull-images", action="store_true", help="whether to pull image")

    #  模型相关参数
    parser.add_argument("--output-path", type=str, default="", help="Path for storing results")
    parser.add_argument("--machine-config", type=str,default="", help="device info file path")
    parser.add_argument("--tasks-config", type=str,nargs='*', default=[], help="JSON file describing the task list")
    parser.add_argument("--user", type=str, default="root", help="client user")
    parser.add_argument("--port",type=int,default=20000,help="client port bind to recv msg")
    parser.add_argument("--incremental-mode", action="store_true", help="only run case not in pass file")
    parser.add_argument("--specify-task", action="store_true", help="Starting from the designated task")
    return parser.parse_args()

def get_ip_from_command():
    try:
        # 执行ip addr命令并获取输出
        result = subprocess.check_output(['ip', 'addr'], text=True)
        
        # 使用正则表达式匹配IPv4地址（排除回环地址）
        ip_pattern = re.compile(r'inet (\d+\.\d+\.\d+\.\d+)/\d+')
        ips = ip_pattern.findall(result)
        
        # 过滤回环地址
        valid_ips = [ip for ip in ips if not ip.startswith('127.')]
        return list(set(valid_ips))  # 去重
    except subprocess.CalledProcessError as e:
        print(f"执行命令失败: {e}")
        return []
    
def check_file_or_dir_is_exist(ip="",username="root",path=""):
    param = "-f"
    if os.path.isdir(path):
        param = "-d"
    cmd = f"ssh {username}@{ip} 'test {param} {path}'"
    result, _ = run_command_realtime(command=cmd,thread_name="check")
    return result

def copy_file_or_dir_to_remote(ip="",username="root",source_path="",target_path=""):
    cmd = f"scp -r {source_path} {username}@{ip}:{target_path}"
    result, _ = run_command_realtime(command=cmd,thread_name="copy")
    return result

def copy_file_or_dir_to_local(ip="",username="root",source_path="",target_path=""):
    cmd = f"scp -r {username}@{ip}:{source_path} {target_path}"
    result, _ = run_command_realtime(command=cmd,thread_name="copy")
    return result

def create_remote_folder(ip="",username="root",path=""):
    cmd = f"ssh {username}@{ip} \"mkdir -p {path}\""
    result, _ = run_command_realtime(command=cmd,thread_name="create_folder")
    print(f"remote folder \"{path}\" create: {result}")
    return result

def get_specified_subfolders_or_file_recursive_os(parent_folder, target_name, is_file=False,is_fuzzy_match=False):
    """
    递归获取指定名字的子文件夹/文件路径列表
    :param parent_folder: 父文件夹路径
    :param target_folder_name: 目标子文件夹/文件名称
    :param is_file: 是否是文件
    :param is_fuzzy_match: 是否模糊匹配
    :return: 符合条件的子文件夹/文件路径列表
    """
    specified_subfolders = []
    for root, dirs, files in os.walk(parent_folder):
        if is_file:
            for file_name in files:
                folder = os.path.dirname(os.path.join(root, file_name)).split("/")[-1]
                if file_name == target_name or f"{folder}/{file_name}" == target_name:
                    specified_subfolders.append(os.path.join(root, file_name))
        else:
            for dir_name in dirs:
                if is_fuzzy_match:
                    if target_name in dir_name:
                        specified_subfolders.append(os.path.join(root, dir_name))
                else:
                    if dir_name == target_name:
                        specified_subfolders.append(os.path.join(root, dir_name))
    return specified_subfolders

def find_files_by_name(root_dir, target_name, recursive=True):
    """
    查找指定目录中名字匹配的文件
    
    参数:
        root_dir: 起始查找目录
        target_name: 要查找的文件名
        recursive: 是否递归查找子目录，默认为True
    返回:
        匹配文件的完整路径列表
    """
    matched_files = []
    
    # 遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_name or target_name in filename:
                # 拼接完整路径
                full_path = os.path.join(dirpath, filename)
                matched_files.append(full_path)
        
        # 如果不递归查找，只处理当前目录就退出
        if not recursive:
            break
    
    return matched_files

def merge_csv_files(csv_paths, output_file, encoding='utf-8', test_type=""):
    """合并CSV文件路径列表中的所有文件"""
    if not csv_paths:
        print("No CSV files were found")
        return

    # 读取并合并所有CSV文件
    dfs = []
    for csv_file in csv_paths:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            dfs.append(df)
        except Exception as e:
            print(f"Read file {csv_file} fail: {str(e)}")
    
    if not dfs:
        print("No CSV files were successfully read")
        return

    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    if  test_type != "acc":
        merged_df = merged_df.sort_values(by=["batch-size"], ascending=True)
    # 保存合并后的结果
    merged_df.to_csv(output_file, index=False, encoding=encoding)
    print(f"The merger is complete {len(merged_df)} line data, saved to: {output_file}")
       
def create_folder(file_dir):
        if os.path.isdir(file_dir) == False:
            try:
                os.makedirs(file_dir, exist_ok=True)
                print(f"folder '{file_dir}' creation successful")
            except FileExistsError:
                print(f"folder '{file_dir}' already exist")
            except OSError as e:
                print(f"creation failed:{e}")
        else:
            print(f"folder '{file_dir}' already exist")
                
def delete_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        try:
            # 递归删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f"folder '{folder_path}' has been successfully deleted")
        except OSError as e:
            print(f"an error occurred when deleting the folder: {e}")
    else:
        print(f"folder '{folder_path}' nonentity")

if __name__ == "__main__":
    Multimachine(parse_args()).run()
   #python3 -m multimachine