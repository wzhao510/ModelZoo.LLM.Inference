#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
business_logic.py - 业务逻辑模块
该模块包含程序的核心业务逻辑，如配置处理、文件复制、模型处理等。
"""

import os
from config_reader import ConfigReader
from content_replacer import ContentReplacer
from file_operations import FileOperations


class BusinessLogic:
    """
    业务逻辑处理器：协调各个模块完成整体流程
    """
    def __init__(self, path_manager):
        self.pm = path_manager
        self.fo = FileOperations(self.pm)
        self.cr = ConfigReader(self.pm, self.fo)
        self.crep = ContentReplacer(self.pm, self.fo)

    def copy_and_replace_configs(self, setting_config: dict, date_folder: str) -> None:
        """
        复制模型配置并执行全量替换
        """
        # 1. 读取映射（现在返回包含target_path和draft_path的嵌套字典）
        model_target_map = self.cr.read_model_csv(run_only=False)
        dataset_map = self.cr.read_dataset_csv()
        
        # 2. 准备目录
        target_configs_dir = os.path.join(date_folder, "configs")
        os.makedirs(target_configs_dir, exist_ok=True)
        print(f"\n📂 【配置复制目标目录】{target_configs_dir}")
        
        # 3. 确定需要复制的模型
        models_root = self.fo.get_abs_path(self.pm.MODELS_ROOT_DIR)
        if not os.path.exists(models_root):
            raise FileNotFoundError(f"❌ 【目录缺失】模型根目录不存在：{models_root}")
        
        # 使用解析后的ALL_RUN_TYPE和CUSTOM_BENCHMARK
        all_run_type = setting_config["ALL_RUN_TYPE"]
        custom_benchmark = setting_config["CUSTOM_BENCHMARK"]
        
        if all_run_type != "custom":
            print(f"🔍 【模式】ALL_RUN={all_run_type}，复制所有模型配置")
            need_copy_models = [
                item for item in os.listdir(models_root)
                if os.path.isdir(os.path.join(models_root, item))
            ]
        else:
            print(f"🔍 【模式】ALL_RUN=custom，仅复制model.csv中RUN=1的模型")
            need_copy_models = self.cr.read_model_csv(run_only=True)
        
        # 4. 复制+替换
        copied_count = 0
        for model_name in need_copy_models:
            src_config = os.path.join(models_root, model_name, "config.json")
            dst_config = os.path.join(target_configs_dir, f"{model_name}_config.json")
            
            if not os.path.exists(src_config):
                print(f"⚠️  [WARNING] 模型{model_name}的config.json不存在：{src_config}，跳过")
                continue
            
            try:
                # 复制文件
                self.fo.copy_file(src_config, dst_config)
                # 替换--model-path（原有逻辑）
                self.crep.replace_server_cmds_model_path(dst_config, model_name, model_target_map)
                # 替换--speculative-draft-model-path（有值则替换，无则跳过）
                self.crep.replace_server_cmds_draft_model_path(dst_config, model_name, model_target_map)
                # 替换ceval--model
                self.crep.replace_ceval_model_path(dst_config, model_target_map[model_name]["target_path"])
                # 替换数据集路径
                self.crep.replace_dataset_paths(dst_config, dataset_map)
                
                # custom:xxx时替换benchmark
                if all_run_type == "custom" and custom_benchmark is not None:
                    print(f"🔄 【Custom模式】替换模型{model_name}的benchmark为：{custom_benchmark}")
                    self.crep.replace_tasks_benchmark(dst_config, custom_benchmark)
                elif all_run_type != "custom":
                    # 非custom模式 → 替换为all_run_type（原有逻辑）
                    self.crep.replace_tasks_benchmark(dst_config, all_run_type)
                # 纯custom（无冒号）→ 不替换，保留原内容
                
                copied_count += 1
            except Exception as e:
                print(f"❌  处理模型{model_name}失败：{str(e)}")
                continue
        
        print(f"\n📊 【复制统计】共处理{len(need_copy_models)}个模型，成功{copied_count}个")

    def process_task_config(self) -> dict:
        """
        处理任务配置（捕获所有配置异常，直接终止）
        """
        print(f"\n📊 【任务配置处理】开始解析setting.csv...")
        # 捕获所有异常（包括校验错误），确保非法格式直接退出
        try:
            setting_config = self.cr.read_setting_csv()
        except Exception as e:
            print(f"\n❌ 【配置解析失败】{str(e)}")
            sys.exit(1)  # 强制退出，绝不继续执行
        
        date_value = setting_config["DATE"]
        config_reuse = setting_config["CONFIG_REUSE"]
        
        # 2. 准备目录
        daily_result_abs = self.fo.get_abs_path(self.pm.DAILY_RESULT_DIR)
        os.makedirs(daily_result_abs, exist_ok=True)
        date_folder = os.path.join(daily_result_abs, date_value)
        os.makedirs(date_folder, exist_ok=True)
        
        print(f"\n📁 【路径解析】")
        print(f"     脚本目录：{os.path.dirname(os.path.abspath(__file__))}")
        print(f"     结果根目录：{daily_result_abs}")
        
        # 3. 备份setting.csv
        setting_src = self.fo.get_abs_path(os.path.join("summary", "setting.csv"))
        setting_dst = os.path.join(date_folder, f"{date_value}_setting.csv")
        
        if config_reuse:
            print(f"\n🔄 【配置复用模式】CONFIG_REUSE=true，仅跳过模型操作")
            self.fo.copy_file(setting_src, setting_dst)
        else:
            print(f"\n🆕 【全新配置模式】CONFIG_REUSE=false，执行清空+备份+复制+替换")
            self.fo.clear_directory(date_folder)
            self.fo.copy_file(setting_src, setting_dst)
            self.copy_and_replace_configs(setting_config, date_folder)
        
        setting_config["DATE_FOLDER_ABS"] = date_folder
        return setting_config