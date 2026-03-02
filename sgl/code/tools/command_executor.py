#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
command_executor.py - 命令执行模块
该模块负责构建和执行Docker启动命令，并处理进程管理和信号处理。
"""

import os
import sys
import subprocess
import signal
import time


class CommandExecutor:
    """
    命令执行器：构建和执行Docker启动命令
    """
    def __init__(self, path_manager, file_ops):
        self.pm = path_manager
        self.fo = file_ops
        self.child_process = None

    def build_docker_cmd(self, setting_config: dict, date_folder: str) -> list:
        """
        构建Docker启动命令
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = ["python3", self.fo.get_abs_path(self.pm.START_SCRIPT)]
        
        # 基础参数
        cmd.extend(["--container-name", setting_config["CONTAINER_NAME"]])
        cmd.extend(["--container-images", setting_config["CONTAINER_IMAGE"]])
        cmd.append("--docker-v")
        cmd.extend(self.pm.DOCKER_VOLUMES)
        
        # 路径参数（转绝对路径）
        target_path_abs = self.fo.get_abs_path(self.pm.TARGET_PATH)
        output_path_abs = self.fo.get_abs_path(self.pm.OUTPUT_PATH)
        machine_config_abs = self.fo.get_abs_path(self.pm.MACHINE_CONFIG)
        tasks_config_abs = os.path.join(date_folder, "configs")
        
        cmd.extend(["--target-path", target_path_abs])
        cmd.extend(["--output-path", output_path_abs])
        cmd.extend(["--tasks-config", tasks_config_abs])

        # 用修改过的machines.json
        date = os.path.basename(date_folder)
        machines_json_path = date_folder + f"/{date}_machines.json"
        cmd.extend(["--machine-config", machines_json_path])
        
        # 布尔参数
        if setting_config["INCREMENTAL_MODE"]:
            cmd.append("--incremental-mode")
        cmd.extend(["--user", setting_config["RUN_USER"]])
        if setting_config["PARALLEL"]:
            cmd.append("--parallel")
        if setting_config["RM_EXIST_DOCKER"]:
            cmd.append("--rm-exist-docker")
        if setting_config["PULL_IMAGES"]:
            cmd.append("--pull-images")
        
        # 其他参数
        cmd.extend(["--local-ip", setting_config["LOCAL_IP"]])
        
        # 打印路径转换日志
        print(f"\n📌 【路径参数转换】")
        print(f"     TARGET_PATH: {target_path_abs}")
        print(f"     TASKS_CONFIG: {tasks_config_abs}")
        return cmd

    def print_cmd_pretty(self, cmd: list) -> None:
        """
        美化打印命令
        """
        print(f"\n📝 【最终执行命令】")
        print("-" * 120)
        print(" ".join(cmd))
        print("\n【分行展示】")
        print("python3")
        for arg in cmd[1:]:
            if arg.startswith("--"):
                print(f"  {arg}")
            else:
                print(f"    {arg}")
        print("-" * 120 + "\n")

    def execute_cmd(self, cmd: list) -> int:
        """
        执行命令并实时输出日志
        """
        self.print_cmd_pretty(cmd)
        
        try:
            print(f"🚀 【开始执行命令】实时日志：")
            time.sleep(3)
            print("-" * 120)
            
            # 修改这里：使用 errors='replace' 或 errors='ignore'
            self.child_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=False, shell=False  # 设置为False，手动处理解码
            )
            
            # 手动读取并解码，处理解码错误
            for raw_line in iter(self.child_process.stdout.readline, b''):
                try:
                    line = raw_line.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    # 如果UTF-8解码失败，尝试其他编码或替换字符
                    try:
                        line = raw_line.decode('gbk', errors='replace')
                    except:
                        # 最后尝试用 Latin-1 解码（不会失败）
                        line = raw_line.decode('latin-1', errors='replace')
                
                print(line, end='')
            
            return_code = self.child_process.wait()
            print("-" * 120)
            
            if return_code == 0:
                print(f"✅ 【执行成功】返回码：{return_code}")
            else:
                print(f"❌ 【执行失败】返回码：{return_code}")
            return return_code
        except Exception as e:
            raise Exception(f"❌ 【执行错误】错误：{str(e)}") from e