#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 程序入口模块
该模块作为程序的入口点，负责初始化程序，加载配置，调用其他模块中的函数或类来启动程序。
"""

import os
import sys
import signal
import json
import subprocess
import concurrent.futures
import time
from config import PathConfig
from business_logic import BusinessLogic
from command_executor import CommandExecutor


class MainProcessor:
    """
    主处理器：协调各个模块完成整体流程
    """
    def __init__(self):
        self.pm = PathConfig()
        self.bl = BusinessLogic(self.pm)
        self.ce = CommandExecutor(self.pm, self.bl.fo)
        self.container_name = None
        self.setting_config = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, sig, frame) -> None:
        """
        Ctrl+C信号处理，停止各个节点上的容器
        """
        print(f"\n⚠️  [WARNING] 收到Ctrl+C中断信号，正在停止所有节点上的容器...")
        
        # 1. 停止本地子进程（如果有）
        if self.ce.child_process and self.ce.child_process.poll() is None:
            try:
                self.ce.child_process.terminate()
                self.ce.child_process.wait(timeout=10)
                print(f"✅  本地子进程已正常退出")
            except subprocess.TimeoutExpired:
                self.ce.child_process.kill()
                print(f"✅  本地子进程已强制终止")
        
        # 2. 停止各个节点上的容器
        self.stop_containers_on_all_nodes()
        
        print(f"✅  所有容器已停止")
        sys.exit(0)

    def stop_containers_on_all_nodes(self) -> None:
        """
        并行停止所有节点上的容器
        """
        # 获取容器名
        container_name = None
        if self.setting_config:
            container_name = self.setting_config.get("CONTAINER_NAME")
        elif self.container_name:
            container_name = self.container_name
        
        if not container_name:
            print(f"⚠️  [WARNING] 无法获取容器名，跳过节点容器停止")
            return
        
        # 读取machines.json获取所有节点IP
        nodes_info = self.get_nodes_from_machines_json()
        if not nodes_info:
            print(f"⚠️  [WARNING] 未找到节点信息，跳过节点容器停止")
            return
        
        print(f"🚀 【并行停止】开始并行停止 {len(nodes_info)} 个节点上的容器 {container_name}...")
        start_time = time.time()
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(nodes_info))) as executor:
            # 提交所有任务
            future_to_node = {
                executor.submit(self.stop_container_on_node_parallel, node, container_name): node
                for node in nodes_info
            }
            
            # 收集结果
            success_count = 0
            fail_count = 0
            
            for future in concurrent.futures.as_completed(future_to_node):
                node = future_to_node[future]
                ip = node.get("ip", "未知IP")
                try:
                    result = future.result()
                    if result:
                        print(f"✅  节点 {ip} 上的容器已停止")
                        success_count += 1
                    else:
                        print(f"⚠️  节点 {ip} 上的容器停止失败")
                        fail_count += 1
                except Exception as e:
                    print(f"❌  节点 {ip} 上的容器停止异常：{str(e)}")
                    fail_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"📊 【停止统计】成功：{success_count}个，失败：{fail_count}个，耗时：{elapsed_time:.2f}秒")

    def stop_container_on_node_parallel(self, node_info: dict, container_name: str) -> bool:
        """
        在指定节点上停止容器（专为并行执行设计）
        """
        ip = node_info.get("ip")
        if not ip:
            print(f"⚠️  [WARNING] 节点信息中缺少IP地址：{node_info}")
            return False
        
        try:
            # 构建SSH命令停止容器
            ssh_cmd = f"ssh {ip} docker stop {container_name} 2>/dev/null || true"
            
            # 设置超时时间
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=20 
            )
            
            if result.returncode == 0:
                # 检查是否真的有容器被停止
                if "Error response from daemon" in result.stderr:
                    # 容器可能不存在
                    return False
                return True
            else:
                print(f"⚠️  节点 {ip} 上的容器停止可能失败：{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌  停止节点 {ip} 上的容器超时")
            return False
        except Exception as e:
            print(f"❌  停止节点 {ip} 上的容器失败：{str(e)}")
            return False
            
    def get_nodes_from_machines_json(self) -> list:
        """
        从machines.json读取所有节点信息
        """
        machines_json_path = self.bl.fo.get_abs_path(self.pm.MACHINE_CONFIG)
        
        if not os.path.exists(machines_json_path):
            print(f"⚠️  [WARNING] machines.json文件不存在：{machines_json_path}")
            return []
        
        try:
            with open(machines_json_path, 'r', encoding='utf-8') as f:
                machines_data = json.load(f)
            
            # 根据提供的JSON结构，节点信息在machine_info数组中
            if "machine_info" in machines_data:
                return machines_data["machine_info"]
            else:
                print(f"⚠️  [WARNING] machines.json中没有找到machine_info字段")
                return []
        except Exception as e:
            print(f"❌  读取machines.json失败：{str(e)}")
            return []

    def main(self) -> None:
        """
        主流程（强化异常捕获）
        """
        print(f"💻 【SGLang Docker启动脚本】")
        print("=" * 120)
        
        try:
            # 切换工作目录到tools/
            os.chdir(self.pm.TOOLS_DIR)
            print(f"📂 【工作目录】{self.pm.TOOLS_DIR}")
            
            # 1. 处理配置（此处会触发所有配置校验）
            self.setting_config = self.bl.process_task_config()
            
            # 保存容器名用于信号处理
            if self.setting_config:
                self.container_name = self.setting_config.get("CONTAINER_NAME")
            
            # 2. 构建命令
            cmd = self.ce.build_docker_cmd(self.setting_config, self.setting_config["DATE_FOLDER_ABS"])
            # 3. 执行命令
            return_code = self.ce.execute_cmd(cmd)
            sys.exit(return_code)
        except Exception as e:
            print(f"\n❌ 【整体流程失败】{str(e)}")
            sys.exit(1)  # 任何异常都终止流程，返回错误码


def main():
    processor = MainProcessor()
    processor.main()


if __name__ == "__main__":
    main()