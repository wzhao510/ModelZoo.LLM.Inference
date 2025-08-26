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

class Machine:

    def __init__(self, config, ip, gpu_count, unused_gpu_count):
        self.config = config
        self.ip = ip
        self.is_local = False
        self.gpu_count = gpu_count
        self.unused_gpu_count = unused_gpu_count
        self.useed_gpu_index = -1

class Multimachine:
    def __init__(self, args:argparse.Namespace):
        self.args:argparse.Namespace = args
        self.total_config = []
        self.machine_list: list[Machine] = []
        self.total_gpu_count = 0
        self.expected_total_gpu_count = 0
        self.total_tasks = []
        self.cancel_cmds = []
        self.master_nodes = []
        

    def run(self) -> None:
        # 注册信号处理函数，监听SIGINT信号（对应Ctrl+C）
        signal.signal(signal.SIGINT, self.handle_ctrl_c)
        local_ip = get_local_ip()
        if local_ip == '0.0.0.0':
            print("Error:## unable to get local ip !")
            exit(1)
        print(f"local ip: {local_ip}")
        
        # 读取测试任务信息
        tasks_path_list = self.args.tasks_config
        for path in tasks_path_list:
            # print(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.total_config.append({"data":data,"path":path})
                tasks = data["tasks"]
                for task in tasks:
                    self.expected_total_gpu_count += self.get_expected_gpu_count(task)
   
        # 读取节点信息   
        with open(self.args.machine_config,"r", encoding="utf-8") as f:
            data = json.load(f)
            for info in data["machine_info"]:
                ip = info["ip"]
                username = self.args.user  # 远程用户名
                count = self.get_gpu_count(ip, username)
                machine = Machine(config=info, ip=ip, gpu_count=count,unused_gpu_count=count)
                machine.is_local = local_ip == ip
                self.machine_list.append(machine)
                self.total_gpu_count += count
            
        print(f"Total of all nodes:{self.total_gpu_count} GPU")
        print(f"Total required for all tasks:{self.expected_total_gpu_count} GPU")
        
        if self.expected_total_gpu_count <= self.total_gpu_count:
            new_json_dir = f"{os.path.dirname(path)}/section"
            delete_folder(new_json_dir)
            server_port = 0
            sgl_port = 3000
            for config in self.total_config:
                new_server_port, new_sgl_port = self.split_config(config["data"],config["path"],server_port,sgl_port)
                server_port = new_server_port
                sgl_port = new_sgl_port
            print(server_port,sgl_port)
            if server_port > 0: 
                self.run_docker(local_ip)
                self.merge_result_data()
        else:
            a = "="
            print(f"{a*35} RUN TASKS {a*35}")
            ip_list = [mc.ip for mc in self.machine_list]
            if not ip_list:
                print("Error: Please configure the machine information !!!")
                return
            tasks_config = " ".join(self.args.tasks_config)
            command = self.create_cmd(self.args.container_name,tasks_config,self.args.machine_config,self.args.port,self.args.output_path)
            if ip_list[0] != local_ip:
                command =  f'ssh -o StrictHostKeyChecking=no {self.args.user}@{ip_list[0]} "cd {current_dir} ; {command}"'
            print(command)
            success, cmd = run_command_realtime(command,"cmd-1")
            status = "succeed" if success else "fail"
            print(f"command: {cmd} -> {status}")
            
    def check_file(self,ip="",username="root",path=""):
        target_path = path
        if check_file_or_dir_is_exist(ip,username,path=target_path) == False:
            create_folder(os.path.dirname(target_path))
            print(target_path, path)
            reslut = copy_file_or_dir_to_remote(ip,username,source_path=path,target_path=target_path)
            print(f"copy config: {reslut}")
        else:
            print(f"file \"{target_path}\" already exist")
                        
    def run_docker(self,local_ip):
    
        port = 0
        all_commands = []
        all_docker_cmd = []
        user = self.args.user      
        for i, task in enumerate(self.total_tasks):
            tasks_config = task["tasks_config_path"]
            machine_config = task["machine_config_path"]
            file_name_with_ext = os.path.basename(tasks_config)
            file_name, _ = os.path.splitext(file_name_with_ext)
            container_name = f"{self.args.container_name}_{file_name}_{i}"
            ip_list = task["ip"]
            if not ip_list:
                print("Error: Please configure the machine information !!!")
                return
            username = self.args.user
            output_path = self.args.output_path
            if port == 0:
                port = self.args.port
            master_node_ip = ip_list[0]
            is_local_machine = master_node_ip == local_ip
            output_path = f"{self.args.output_path}/multi"
            node = {"ip":master_node_ip,"is_local":is_local_machine,"path":output_path,"task_type":file_name}
            if is_local_machine:
                self.master_nodes.insert(0,node)
            else:
                self.master_nodes.append(node)
            cmd = self.create_cmd(container_name, tasks_config, machine_config, port, output_path)
            if is_local_machine:
                create_folder(output_path)
            else:
                create_remote_folder(master_node_ip,self.args.user,output_path)        
                cmd =  f'ssh -o StrictHostKeyChecking=no {user}@{master_node_ip} "cd {current_dir} ; {cmd}"'

            port +=10000
            docker_cmd = f"docker rm {container_name}"
            docker_stop_cmd = f"docker stop {container_name}"
            for ip in ip_list:
                if ip != local_ip:
                    docker_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "{docker_cmd}"'
                    docker_stop_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "{docker_stop_cmd}"'
                    self.check_file(ip,username,tasks_config)
                    self.check_file(ip,username,machine_config)
                    if check_file_or_dir_is_exist(ip,username,output_path) == False:
                        create_folder(output_path)
                all_docker_cmd.append(docker_cmd)
                self.cancel_cmds.append(docker_stop_cmd)
                self.cancel_cmds.append(docker_cmd)

            all_commands.append(cmd)
           
        # print(all_commands)
        # print(all_docker_cmd)
            
        print("Start concurrent execution of commands...")
    

        task_run_results = run_commands_concurrently(
            commands=all_commands,
            max_workers=len(all_commands)  # 限制最大并发数
        )

        # 输出最终执行结果
        print("\n===== The test task command has been executed successfully =====")
        for cmd, success in task_run_results.items():
            status = "succeed" if success else "fail"
            print(f"command: {cmd} -> {status}")
   
        docker_rm_results = run_commands_concurrently(
            commands=all_docker_cmd,
            max_workers=len(all_docker_cmd)  # 限制最大并发数
        )
        
        print("\n===== All commands have been executed =====")
        for cmd, success in docker_rm_results.items():
            status = "succeed" if success else "fail"
            print(f"command: {cmd} -> {status}")
         
    def merge_result_data(self):
        # %Y%m%d_%H%M%S
        time_str = time.strftime("%Y%m%d_%H", time.localtime())
        # time_str = "20250825_141600"
        print(time_str)
        dst_path = ""
        task_types = set()
        for node in self.master_nodes:
            is_local = node["is_local"]
            task_type = node["task_type"]
            task_types.add(task_type)
            path = node["path"]
            if is_local == False:
                copy_file_or_dir_to_local(node["ip"],self.args.user,path,dst_path)
            else:
                dir = os.path.dirname(path)
                dst_path = f"{dir}/{time_str}"
                create_folder(dst_path)
                print(dst_path)
                
            print(dst_path, task_type)    
            result = get_specified_subfolders_recursive_os(dst_path, task_type)
            # print(result)
            for p in result:
                # rsync -av --progress
                cmd = f"rsync -av {p} {dst_path}" 
                run_command_realtime(cmd,thread_name="rsync")
            delete_folder(f"{dst_path}/multi")
        
        name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        for type in task_types:
            result = find_files_by_name(f"{dst_path}/{type}",f"{type}_result.csv")
            first = os.path.dirname(result[0])
            output_path = os.path.dirname(first)
            for f in find_files_by_name(output_path,"_result.csv",False):
                os.remove(f)
            merge_csv_files(result,f"{output_path}/{name}_result.csv")            
       
    def split_config(self,data,path,server_port,sgl_port):
        """
        切分任务
        """
        a = "=="
        # 读取task信息
        tasks = data["tasks"]
        if self.expected_total_gpu_count <= self.total_gpu_count:
            new_json_dir = f"{os.path.dirname(path)}/section"
            create_folder(new_json_dir)
            file_name_with_ext = os.path.basename(path)
            file_name, ext = os.path.splitext(file_name_with_ext)
            
            for i, value in enumerate(tasks):
                
                if server_port == 0:
                    server_port = int(value["server_port"])
                machine_info = []
                use_machine = None
                expected_gpu_count = self.get_expected_gpu_count(value)
                print(f"\n{a*35} START {a*35}")
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
                        
                new_json_dir = f"{os.path.dirname(path)}/section/{file_name}_{i}"
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
                    self.total_tasks.append({"tasks_config_path":new_json_path, "machine_config_path":machine_config_path,"ip":[machine["ip"] for machine in machine_info]})
                    sgl_port +=10
                    server_port += 1000
                print(f"{a*35} END {a*35}\n")   
            return server_port, sgl_port
        else:
            print("The number of cards required exceeds the actual number of available cards")
            return 0,0

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

        if task["launch_mode"] == "offline":
            command = "".join(task["server_base"]["param"])
            tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
            return int(tp_size_match.group(2))
        else:
            command = "".join(task["launch_server"]["enable_parallel"])
            tp_size_match = re.search(r"--(tp|tp-size)\s+(\d+)",command)
            return  int(tp_size_match.group(2))
            
    def get_gpu_count(self,ip, username, port=22): 
        #  "mx-smi --show-pcie" 
        gpu_count = -1
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
           success, _ = run_command_realtime(command=cmd,thread_name="cancel")
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
    
    # 线程池执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                run_command_realtime, 
                cmd, 
                thread_name=f"cmd-{i+1}"  # 为每个命令指定线程名
            ): cmd 
            for i, cmd in enumerate(commands)
        }
        
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


def get_local_ip():
    """
    获取本机的IP地址
    返回值: 本机的IP地址字符串，如果获取失败则返回None
    """
    try:
        # 创建一个socket连接来确定本机IP
        # 这里连接的地址不需要实际可达，只是为了获取当前机器的出口IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 使用公共DNS服务器地址，不实际建立连接
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception as e:
        print(f"An error occurred when obtaining the IP address: {e}")
        return None
    
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
    print(f"folder \"{path}\" create: {result}")
    return result

def get_specified_subfolders_recursive_os(parent_folder, target_folder_name):
    """
    递归获取指定名字的子文件夹路径列表
    :param parent_folder: 父文件夹路径
    :param target_folder_name: 目标子文件夹名称
    :return: 符合条件的子文件夹路径列表
    """
    specified_subfolders = []
    for root, dirs, files in os.walk(parent_folder):
        for dir_name in dirs:
            if dir_name == target_folder_name:
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

def merge_csv_files(csv_paths, output_file, encoding='utf-8'):
    """合并CSV文件路径列表中的所有文件"""
    if not csv_paths:
        print("No CSV files were found")
        return

    # 读取并合并所有CSV文件
    dfs = []
    for csv_file in csv_paths:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            # 可选：添加一列记录数据来源文件
            df['source_file'] = os.path.basename(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Read file {csv_file} fail: {str(e)}")
    
    if not dfs:
        print("No CSV files were successfully read")
        return

    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
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
    # TODO 兼容没有共享目录（已完成）；测试结果合并：output 添加标识，测试结束后遍历合并
   #python3 -m multimachine
   