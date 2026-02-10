#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_replacer.py - 内容替换模块
该模块负责处理JSON配置文件中的内容替换操作，如路径替换、参数更新等。
"""
import json
import re
from config import ValidationConfig


class ContentReplacer:
    """
    内容替换器：负责JSON配置文件的内容替换
    """
    def __init__(self, path_manager, file_ops):
        self.pm = path_manager
        self.fo = file_ops
        self.validation_config = ValidationConfig()

    def replace_json_block_content(self, content: str, block_key: str, old_pattern: str, new_val: str) -> str:
        """
        通用：替换JSON指定块内的内容
        """
        block_content, start_idx, end_idx = self.fo.get_json_block(content, block_key)
        if start_idx == -1:
            print(f"⚠️  [WARNING] 未找到{block_key}块，跳过替换")
            return content
        
        new_block_content, replace_count = re.subn(
            old_pattern,
            lambda m: f"{m.group(1)} {new_val}",
            block_content
        )
        
        if replace_count == 0:
            print(f"⚠️  [WARNING] {block_key}块内未找到匹配内容，跳过替换")
            return content
        
        new_content = content[:start_idx] + new_block_content + content[end_idx+1:]
        print(f"✅  替换{block_key}块 → {replace_count}处替换为：{new_val}")
        return new_content

    def replace_server_cmds_model_path(self, file_path: str, model_name: str, model_target_map: dict) -> None:
        """
        替换server_cmds中的--model-path
        """
        if model_name not in model_target_map or "target_path" not in model_target_map[model_name]:
            raise ValueError(f"❌ 【错误】model.csv中无模型{model_name}的TARGET_PATH")
        
        target_path = model_target_map[model_name]["target_path"]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = self.replace_json_block_content(content, "server_cmds", 
                                                          self.validation_config.MODEL_PATH_PATTERN, target_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            raise Exception(f"❌  替换{file_path}的--model-path失败：{str(e)}") from e

    def replace_server_cmds_draft_model_path(self, file_path: str, model_name: str, model_target_map: dict) -> None:
        """
        替换server_cmds中的--speculative-draft-model-path（有值则替换，无则跳过）
        """
        if model_name not in model_target_map:
            raise ValueError(f"❌ 【错误】model.csv中无模型{model_name}")
        
        draft_path = model_target_map[model_name].get("draft_path")
        # 核心修改：draft_path为None/空时直接跳过，不报错
        if not draft_path:
            print(f"⚠️  [WARNING] 模型{model_name}无DRAFT_PATH，跳过--speculative-draft-model-path替换")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = self.replace_json_block_content(
                content, "server_cmds", self.validation_config.SPECULATIVE_DRAFT_MODEL_PATTERN, draft_path
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            raise Exception(f"❌  替换{file_path}的--speculative-draft-model-path失败：{str(e)}") from e

    def replace_ceval_model_path(self, file_path: str, target_path: str) -> None:
        """
        替换benchmark_cmds.ceval中的--model
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            bench_block, bench_start, bench_end = self.fo.get_json_block(content, "benchmark_cmds")
            if bench_start == -1:
                print(f"⚠️  [WARNING] {file_path}无benchmark_cmds，跳过--model替换")
                return
            
            ceval_block, ceval_start, ceval_end = self.fo.get_json_sub_block(bench_block, "ceval")
            if ceval_start == -1:
                print(f"⚠️  [WARNING] {file_path}无ceval子块，跳过--model替换")
                return
            
            new_ceval_block, replace_count = re.subn(
                self.validation_config.CEVAL_MODEL_PATTERN,
                lambda m: f"{m.group(1)} {target_path}",
                ceval_block
            )
            if replace_count == 0:
                print(f"⚠️  [WARNING] {file_path}的ceval块内未找到--model，跳过替换")
                return
            
            new_bench_block = bench_block[:ceval_start] + new_ceval_block + bench_block[ceval_end+1:]
            new_content = content[:bench_start] + new_bench_block + content[bench_end+1:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅  {file_path} → ceval--model替换为：{target_path}")
        except Exception as e:
            raise Exception(f"❌  替换{file_path}的ceval--model失败：{str(e)}") from e

    def replace_dataset_paths(self, file_path: str, dataset_map: dict) -> None:
        """
        替换benchmark_cmds中各数据集的路径，并添加额外配置
        简化版：直接修改整个JSON结构
        """
        try:
            # 读取整个JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否存在benchmark_cmds
            if "benchmark_cmds" not in data:
                print(f"⚠️  [WARNING] {file_path}无benchmark_cmds，跳过数据集路径替换")
                return
            
            benchmark_cmds = data["benchmark_cmds"]
            
            for dataset_name in ["random", "ceval", "mmlu"]:
                dataset_lower = dataset_name.lower()
                
                # 检查配置中是否有该数据集
                if dataset_name not in benchmark_cmds:
                    print(f"⚠️  [WARNING] {file_path}无{dataset_name}配置，跳过")
                    continue
                
                # 检查dataset.csv中是否有该数据集配置
                if dataset_lower not in dataset_map:
                    print(f"⚠️  [WARNING] dataset.csv中无{dataset_name}的配置，跳过")
                    continue
                
                dataset_info = dataset_map[dataset_lower]
                target_path = dataset_info["path"]
                extra_config = dataset_info.get("value")
                
                # 1. 替换路径
                ds_config = benchmark_cmds[dataset_name]
                
                # 检查是否有command_base字段（mmlu使用这种方式）
                if "command_base" in ds_config and isinstance(ds_config["command_base"], str):
                    param_name = self.validation_config.DATASET_PARAM_MAP.get(dataset_lower, "")
                    if param_name:
                        # 使用正则替换路径
                        command_base = ds_config["command_base"]
                        pattern = rf'({re.escape(param_name)})\s+/[^"\s]+'
                        new_command_base, count = re.subn(
                            pattern,
                            f'\\1 {target_path}',
                            command_base
                        )
                        if count > 0:
                            ds_config["command_base"] = new_command_base
                            print(f"✅  {file_path} → {dataset_name}.{param_name}替换为：{target_path}")
                
                # 2. 添加额外配置（如果存在）
                if extra_config is not None:
                    # 检查extra_config是否是字典
                    if isinstance(extra_config, dict):
                        # 合并配置
                        for key, value in extra_config.items():
                            ds_config[key] = value
                        print(f"✅  {file_path} → {dataset_name}添加额外配置：{extra_config}")
                    else:
                        # 如果不是字典，可能是其他类型（如列表、字符串、数字等）
                        # 我们可以选择存储到一个特殊字段中，或者打印警告
                        print(f"⚠️  [WARNING] {dataset_name}的VALUE不是字典类型，跳过配置添加：{extra_config}")
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅  {file_path} → 数据集路径和配置更新完成")
        except Exception as e:
            raise Exception(f"❌  替换{file_path}数据集路径失败：{str(e)}") from e
            
    def replace_tasks_benchmark(self, file_path: str, new_benchmark: str) -> None:
        """
        替换tasks块内的所有benchmark值
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tasks_block, start_idx, end_idx = self.fo.get_json_block(content, "tasks")
            if start_idx == -1:
                print(f"⚠️  [WARNING] {file_path}无tasks块，跳过benchmark替换")
                return
            
            benchmark_pattern = r'"benchmark"\s*:\s*"([^"]*)"'
            new_tasks_block, replace_count = re.subn(
                benchmark_pattern,
                f'"benchmark": "{new_benchmark}"',
                tasks_block
            )
            
            if replace_count == 0:
                print(f"⚠️  [WARNING] {file_path}的tasks块内未找到benchmark，跳过")
                return
            
            new_content = content[:start_idx] + new_tasks_block + content[end_idx+1:]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅  {file_path} → {replace_count}处benchmark替换为：{new_benchmark}")
        except Exception as e:
            raise Exception(f"❌  替换{file_path}的benchmark失败：{str(e)}") from e