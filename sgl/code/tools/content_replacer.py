#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_replacer.py - 内容替换模块
该模块负责处理JSON配置文件中的内容替换操作，如路径替换、参数更新等。
"""
import re
import ast
import json
import pandas as pd
from config import ValidationConfig, PathConfig


class ContentReplacer:
    """
    内容替换器：负责JSON配置文件的内容替换
    """
    def __init__(self, path_manager, file_ops):
        self.pm = path_manager
        self.fo = file_ops
        self.validation_config = ValidationConfig()

    def replace_model_dataset_path(self, file_path: str, pathconfig: PathConfig) -> None:
        """
        替换machines.json中的内容
        """
        # 读取整个JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        # 检查是否存在benchmark_cmds
        if "replacements" not in data:
            raise ValueError(f"❌ {file_path} 中没有 replacements")

        # 读取 CSV 文件
        # path_dict = {}
        root_csv_path = pathconfig.CSV_ROOT_DIR

        dataset_csv_path = root_csv_path + f"/dataset.csv"
        dataset_csv = pd.read_csv(dataset_csv_path)
        dataset_dict = dict(zip(
                            dataset_csv["DATASET"], 
                            dataset_csv[["PATH"]].values.tolist()  # 把 PATH、TYPE 打包成列表
                        ))
        # path_dict.update(dataset_dict)

        model_csv_path = root_csv_path + f"/model.csv"
        model_csv = pd.read_csv(model_csv_path)
        model_dict = dict(zip(
                            model_csv["MODEL"], 
                            model_csv[["TARGET_PATH","DRAFT_PATH"]].values.tolist()  # 把 PATH、TYPE 打包成列表
                        ))
        # path_dict.update(model_dict)

        self.replace_dataset_paths(file_path, dataset_dict)
        self.replace_model_paths(file_path, model_dict)

    def replace_dataset_paths(self, file_path, dataset_dict, fixed_suffix="-dataset-path"):
        """
        精准匹配：键名（如random-dataset-path）必被双引号包裹，值也必被双引号包裹
        格式要求："xxx-dataset-path": "value", （单行、冒号、逗号结尾）
        """
        try:
            # 1. 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            replace_count = 0  # 统计成功替换次数

            def replace_match(match):
                nonlocal replace_count
                # 提取匹配到的关键部分
                full_key = match.group(1)       # 例如：random-dataset-path
                original_value = match.group(2) # 例如：/models/old_random_path.json
                
                # 提取键名前缀（如 random-dataset-path → random）
                key_prefix = full_key.replace(fixed_suffix, "")
                
                # 检查是否在替换字典中且值有效
                if key_prefix in dataset_dict and isinstance(dataset_dict[key_prefix], list) and len(dataset_dict[key_prefix]) > 0:
                    new_value = dataset_dict[key_prefix][0]
                    replace_count += 1
                    # 打印匹配和替换信息
                    # print(f"🔍 匹配到：\"{full_key}\": \"{original_value}\"")
                    # print(f"✅ 替换为：\"{full_key}\": \"{new_value}\"")
                    # 返回替换后的完整内容（保留双引号格式）
                    return f'"{full_key}": "{new_value}"'
                
                # 未匹配到字典则返回原内容
                return match.group(0)

            # 2. 构造精准正则：仅匹配 "键名": "值" 格式
            # 正则说明：
            # "([^"]+{fixed_suffix})"  → 匹配双引号包裹的键名（如 "random-dataset-path"）
            # :\s*                     → 匹配冒号 + 任意空格
            # "([^"]+)"                → 匹配双引号包裹的值（值内无双引号）
            pattern = re.compile(
                rf'"([^"]+{re.escape(fixed_suffix)})":\s*"([^"]+)"',
                re.MULTILINE  # 按行匹配，确保单行处理
            )

            # 3. 执行全局替换
            updated_content = pattern.sub(replace_match, file_content)

            # 4. 写回原文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            # 打印最终统计
            print(f"\n📊 替换完成：共扫描到 {len(pattern.findall(file_content))} 个匹配项，成功替换 {replace_count} 处")
            print(f"✅ 文件已保存：{file_path}")

        except FileNotFoundError:
            print(f"❌ 错误：文件 {file_path} 不存在，请检查路径")
        except PermissionError:
            print(f"❌ 错误：无权限读写文件 {file_path}")
        except Exception as e:
            print(f"❌ 执行失败：{str(e)}")


    def replace_model_paths(self, file_path, model_dict):
        """
        替换文件中模型路径：
        - 用 model_dict[key][0] 替换 "key-model-path" 的值
        - 若 model_dict[key][1] 不是 "nan" 字符串，则用它替换 "key-draft-model-path" 的值
        """
        try:
            # 1. 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            replace_count = 0

            def replace_match(match):
                nonlocal replace_count
                full_key = match.group(1)  # 如 "DeepSeek-R1-AWQ-model-path"
                original_value = match.group(2)

                # 从完整键名中提取前缀和类型（model-path / draft-model-path）
                if "-draft-model-path" in full_key:
                    key_prefix = full_key.replace("-draft-model-path", "")
                    path_type = "draft"
                elif "-model-path" in full_key:
                    key_prefix = full_key.replace("-model-path", "")
                    path_type = "normal"
                else:
                    return match.group(0)

                # 检查前缀是否在字典中
                if key_prefix not in model_dict:
                    return match.group(0)

                value_list = model_dict[key_prefix]
                if not isinstance(value_list, list) or len(value_list) < 2:
                    return match.group(0)

                # 根据路径类型决定替换值
                if path_type == "normal":
                    new_value = value_list[0]
                    replace_count += 1
                    # print(f"🔍 匹配: \"{full_key}\": \"{original_value}\"")
                    # print(f"✅ 替换: \"{full_key}\": \"{new_value}\"")
                    return f'"{full_key}": "{new_value}"'
                elif path_type == "draft":
                    # 关键修改：判断是否为字符串 "nan"
                    if str(value_list[1]).strip().lower() != "nan":
                        new_value = value_list[1]
                        replace_count += 1
                        # print(f"🔍 匹配: \"{full_key}\": \"{original_value}\"")
                        # print(f"✅ 替换: \"{full_key}\": \"{new_value}\"")
                        return f'"{full_key}": "{new_value}"'
                    else:
                        # print(f"ℹ️ 跳过: \"{full_key}\"，对应值为nan字符串")
                        return match.group(0)

                return match.group(0)

            # 2. 精准正则：匹配 "xxx-model-path": "value" 或 "xxx-draft-model-path": "value"
            pattern = re.compile(
                r'"([^"]+-(draft-)?model-path)":\s*"([^"]+)"',
                re.MULTILINE
            )

            # 3. 执行替换
            updated_content = pattern.sub(replace_match, file_content)

            # 4. 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            print(f"\n✅ 完成：共成功替换 {replace_count} 处路径")
            return True

        except Exception as e:
            print(f"❌ 错误：{str(e)}")
            import traceback
            traceback.print_exc()
            return False

    
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
            
            def process_benchmark(match):
                # 1. 提取原始 benchmark 值并拆分
                original_val = match.group(1)
                original_items = [item.strip() for item in original_val.split(';') if item.strip()]
                
                do_acc = "ceval" in benchmark_item_list or "mmlu" in benchmark_item_list
                do_perf = "random" in benchmark_item_list
                do_acc_only = "random" not in benchmark_item_list and do_acc
                do_perf_only = "random" not in benchmark_item_list and not do_acc
                do_both = do_acc and do_perf
                # 精度测试的时候不测 “仅维持功能正常” 的特性（task前面会标注[verify]）
                if original_items == ["verify"] and do_acc_only:
                    return f'"benchmark": ""'
                if original_items == ["verify"] and do_both:
                    return f'"benchmark": "verify"'
                # 2. 新增核心逻辑：过滤原始值，只保留在合法列表中的项
                # （删除原始值中不在 benchmark_item_list 里的内容）
                filtered_items = [item for item in original_items if (item in benchmark_item_list or item == "verify")]
                
                # 3. 保留原有逻辑：检查是否包含 verify，若包含则排除 random
                exclude_random = "verify" in filtered_items  # 基于过滤后的值判断
                
                # 4. 遍历合法列表，按规则追加（去重 + 排除 random）
                final_items = filtered_items.copy()

                for task_type in benchmark_item_list:
                    # 如果需要排除random，且当前task_type是random，则跳过
                    if exclude_random and task_type == "random":
                        continue
                    # 确保不重复添加
                    if task_type not in final_items:
                        final_items.append(task_type)
                
                # 5. 合并为新的 benchmark 字符串并返回
                new_val = ';'.join(final_items)
                return f'"benchmark": "{new_val}"'

            def remove_empty_benchmark_blocks(tasks_block):
                # 1. 先清理前后空白
                tasks_block = tasks_block.strip()

                # 2. 如果字符串以 "tasks": 开头，说明缺少外层 {}
                if tasks_block.startswith('"tasks":'):
                    # 补全外层大括号
                    fixed_block = '{' + tasks_block + '}'
                else:
                    fixed_block = tasks_block

                # 3. 现在尝试解析
                try:
                    data = ast.literal_eval(fixed_block)
                    print("✅ 解析成功！")
                    # 继续你的过滤逻辑
                    if "tasks" in data and isinstance(data["tasks"], dict):
                        data["tasks"] = {
                            task_name: task_config
                            for task_name, task_config in data["tasks"].items()
                            if task_config.get("benchmark", "").strip() != ""
                        }
                    # 转回字符串
                    cleaned_block = json.dumps(data, indent=2) 
                    s = cleaned_block.strip("'")
                    # 2. 再通过切片去除首尾的大括号
                    result = s[1:-1]
                    return result
                except Exception as e:
                    print(f"❌ 解析失败: {e}")

            # 3. 执行替换
            benchmark_item_list = [item.strip() for item in new_benchmark.split(';') if item.strip()]
            benchmark_pattern = r'"benchmark"\s*:\s*"([^"]*)"'
            new_tasks_block, replace_count = re.subn(benchmark_pattern, process_benchmark, tasks_block)
            filtered_tasks_block = remove_empty_benchmark_blocks(new_tasks_block)
            if replace_count == 0:
                print(f"⚠️  [WARNING] {file_path}的tasks块内未找到benchmark，跳过")
                return
            
            new_content = content[:start_idx] + filtered_tasks_block + content[end_idx+1:]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅  {file_path} → {replace_count}处benchmark替换为：{new_benchmark}")
        except Exception as e:
            raise Exception(f"❌  替换{file_path}的benchmark失败：{str(e)}") from e

    def replace_dataset_paths_in_benchmark_cmds(self, file_path: str, dataset_map: dict) -> None:
        """
        添加额外配置
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
            
            for dataset_name in ["random", "ceval", "mmlu", "verify"]:
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
                extra_config = dataset_info.get("value")
                
                # 添加额外配置（如果存在）
                ds_config = benchmark_cmds[dataset_name]
                
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
