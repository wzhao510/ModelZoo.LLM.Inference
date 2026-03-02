#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_reader.py - 配置读取模块
该模块负责读取和解析CSV配置文件，提供配置验证和数据转换功能。
"""

import os
import json
import csv
from config import ValidationConfig


class ConfigReader:
    """
    配置读取器：负责读取CSV配置文件
    """
    def __init__(self, path_manager, file_ops):
        self.pm = path_manager
        self.fo = file_ops
        self.validation_config = ValidationConfig()

    def read_csv_to_map(self, csv_filename: str, key_col: str, val_col: str, lower_key: bool = False) -> dict:
        """
        通用：读取CSV为{key: value}映射
        """
        csv_path = os.path.join(self.pm.CSV_ROOT_DIR, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 【文件缺失】未找到{csv_filename}，路径：{csv_path}")
        
        result_map = {}
        try:
            with open(csv_path, "r", encoding="utf-8", newline='') as f:
                reader = csv.DictReader(f)
                header_map = {h.lower(): h for h in reader.fieldnames}
                
                # 校验必须列
                for col in [key_col, val_col]:
                    if col.lower() not in header_map:
                        raise ValueError(
                            f"❌ 【表头错误】{csv_filename}必须包含{col.upper()}列（大小写不敏感）\n"
                            f"📋 当前表头：{reader.fieldnames}"
                        )
                
                # 构建映射
                key_col_real = header_map[key_col.lower()]
                val_col_real = header_map[val_col.lower()]
                for line_num, row in enumerate(reader, 2):
                    key = row[key_col_real].strip()
                    val = row[val_col_real].strip()
                    
                    if not key:
                        print(f"⚠️  [WARNING] {csv_filename}第{line_num}行{key_col}列为空，跳过")
                        continue
                    if not val:
                        print(f"⚠️  [WARNING] {csv_filename}第{line_num}行{val_col}为空，{key} → 跳过")
                        continue
                    
                    if lower_key:
                        key = key.lower()
                    result_map[key] = val
                    print(f"📝 【{csv_filename}行{line_num}】{key} → {val_col}={val}")
            
            print(f"\n✅ 【{csv_filename}映射】共加载{len(result_map)}条数据")
            return result_map
        except Exception as e:
            raise Exception(f"❌  读取{csv_filename}失败：{str(e)}") from e

    def read_setting_csv(self) -> dict:
        """
        读取setting.csv配置
        """
        setting_map = self.read_csv_to_map("setting.csv", "setting", "value")
        
        missing_fields = [f for f in self.validation_config.SETTING_REQUIRED_FIELDS if f not in setting_map]
        if missing_fields:
            raise ValueError(f"❌ 【配置缺失】setting.csv缺少必须字段：{missing_fields}")
        
        for field in self.validation_config.SETTING_BOOL_FIELDS:
            val = setting_map[field].lower()
            if val in ["true", "1", "是", "yes"]:
                setting_map[field] = True
            elif val in ["false", "0", "否", "no"]:
                setting_map[field] = False
            else:
                raise ValueError(f"❌ 【配置错误】{field}值无效，必须是True/False/1/0，当前值：{setting_map[field]}")
        
        # 解析ALL_RUN配置
        all_run_val = setting_map["ALL_RUN"].strip()
        setting_map["ALL_RUN_RAW"] = all_run_val
        setting_map["ALL_RUN_TYPE"] = None
        setting_map["CUSTOM_BENCHMARK"] = None
        
        if ":" in all_run_val:
            # 按第一个冒号拆分“类型:基准值”
            run_type_part, benchmark_part = all_run_val.split(":", 1)
            run_type = run_type_part.strip().lower()
            benchmark_str = benchmark_part.strip()
            
            setting_map["ALL_RUN_TYPE"] = run_type
            setting_map["CUSTOM_BENCHMARK"] = benchmark_str
            
            # 仅对 custom 类型做强制分隔符校验
            if run_type == "custom":
                # 检查是否包含任何非法分隔符
                has_illegal = any(char in benchmark_str for char in self.validation_config.ILLEGAL_SEPARATORS)
                if has_illegal:
                    # 收集所有存在的非法字符
                    found_illegal = [char for char in self.validation_config.ILLEGAL_SEPARATORS if char in benchmark_str]
                    error_msg = (
                        f"\n❌ 【配置错误】ALL_RUN 格式无效：{all_run_val}\n"
                        f"   检测到非法分隔符：{found_illegal}（仅允许使用 半角分号 ; 分隔）\n"
                        f"   正确示例：custom:random;ceval"
                    )
                    print(error_msg)
                    raise ValueError(error_msg)  # 抛出异常，直接终止流程
                
                # 可选：校验基准值是否合法（仅警告，不终止）
                if ";" in benchmark_str:
                    benchmark_items = [item.strip().lower() for item in benchmark_str.split(";") if item.strip()]
                else:
                    benchmark_items = [benchmark_str.strip().lower()]
                
                invalid_items = [item for item in benchmark_items if item not in self.validation_config.VALID_BENCHMARKS]
                if invalid_items:
                    print(f"⚠️  [警告] ALL_RUN 包含无效基准值：{invalid_items}，建议仅使用 {self.validation_config.VALID_BENCHMARKS}")
        else:
            # 无冒号：直接作为类型（如 custom/ceval/random）
            setting_map["ALL_RUN_TYPE"] = all_run_val.strip().lower()
            setting_map["CUSTOM_BENCHMARK"] = None
        
        setting_map["CONTAINER_NAME"] = f"daily_test_{setting_map['DATE']}"
        print(f"\n✅ 【setting.csv读取成功】")
        print(f"   ALL_RUN原始值：{setting_map['ALL_RUN_RAW']}")
        print(f"   ALL_RUN类型：{setting_map['ALL_RUN_TYPE']}")
        print(f"   自定义基准值：{setting_map['CUSTOM_BENCHMARK']}")
        return setting_map

    def read_model_csv(self, run_only: bool = False):
        """
        读取model.csv配置
        :param run_only: True=仅返回RUN=1的模型名，False=返回映射字典
        :return: 映射字典或模型名列表
        """
        csv_path = os.path.join(self.pm.CSV_ROOT_DIR, "model.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 【文件缺失】未找到model.csv，路径：{csv_path}")
        
        if run_only:
            # 仅按RUN=1筛选，不管draft_path是否存在
            run_models = []
            with open(csv_path, "r", encoding="utf-8", newline='') as f:
                reader = csv.DictReader(f)
                header_map = {h.lower(): h for h in reader.fieldnames}
                if "run" not in header_map or "model" not in header_map:
                    raise ValueError(f"❌ 【表头错误】model.csv必须包含RUN和MODEL列")
                
                run_col = header_map["run"]
                model_col = header_map["model"]
                for line_num, row in enumerate(reader, 2):
                    run_val = row[run_col].strip()
                    model_name = row[model_col].strip()
                    
                    if not model_name:
                        print(f"⚠️  [WARNING] model.csv第{line_num}行MODEL列为空，跳过")
                        continue
                    
                    # 仅判断RUN=1，不校验任何路径字段
                    if run_val == "1":
                        run_models.append(model_name)
                        print(f"📝 【model.csv行{line_num}】模型{model_name} → 执行（RUN={run_val}）")
                    else:
                        print(f"📝 【model.csv行{line_num}】模型{model_name} → 跳过（RUN={run_val}）")
            
            print(f"\n✅ 【model.csv读取成功】需执行模型：{run_models}")
            return run_models
        else:
            # 仅强制model+target_path，draft_path为空则设为None，不跳过模型
            model_map = {}
            with open(csv_path, "r", encoding="utf-8", newline='') as f:
                reader = csv.DictReader(f)
                header_map = {h.lower(): h for h in reader.fieldnames}
                
                # 校验必须列：仅强制model+target_path，draft_path可选
                required_cols = ["model", "target_path"]
                missing_cols = [col for col in required_cols if col.lower() not in header_map]
                if missing_cols:
                    raise ValueError(
                        f"❌ 【表头错误】model.csv必须包含{', '.join(required_cols)}列（大小写不敏感）\n"
                        f"📋 当前表头：{reader.fieldnames}"
                    )
                
                # 可选列：draft_path
                has_draft_col = "draft_path" in header_map
                model_col = header_map["model"]
                target_col = header_map["target_path"]
                draft_col = header_map["draft_path"] if has_draft_col else None
                
                for line_num, row in enumerate(reader, 2):
                    model_name = row[model_col].strip()
                    target_path = row[target_col].strip()
                    
                    # 安全读取draft_path
                    if has_draft_col and draft_col:
                        draft_value = row.get(draft_col, "")  # 使用get避免KeyError
                        if draft_value is None:
                            draft_path = None
                        else:
                            draft_path = draft_value.strip()
                            if not draft_path:
                                draft_path = None
                    else:
                        draft_path = None
                    
                    # 仅校验model+target_path非空
                    if not model_name:
                        print(f"⚠️  [WARNING] model.csv第{line_num}行MODEL列为空，跳过")
                        continue
                    if not target_path:
                        print(f"⚠️  [WARNING] model.csv第{line_num}行TARGET_PATH为空，{model_name} → 跳过")
                        continue
                    
                    model_map[model_name] = {
                        "target_path": target_path,
                        "draft_path": draft_path
                    }
                    draft_log = draft_path if draft_path else "None"
                    print(f"📝 【model.csv行{line_num}】{model_name} → TARGET_PATH={target_path}, DRAFT_PATH={draft_log}")
            
            print(f"\n✅ 【model.csv映射】共加载{len(model_map)}条数据（draft_path可选）")
            return model_map

    def read_dataset_csv(self) -> dict:
        """
        读取dataset.csv配置（键转小写）
        返回结构：{
            "random": {
                "path": "/path/to/dataset",
                "value": {"max_concurrency": [...], "num_prompt_times": 1}
            },
            ...
        }
        """
        csv_path = os.path.join(self.pm.CSV_ROOT_DIR, "dataset.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 【文件缺失】未找到dataset.csv，路径：{csv_path}")
        
        dataset_map = {}
        try:
            with open(csv_path, "r", encoding="utf-8", newline='') as f:
                reader = csv.DictReader(f)
                header_map = {h.lower(): h for h in reader.fieldnames}
                
                # 必须列校验
                required_cols = ["dataset", "path"]
                missing_cols = [col for col in required_cols if col.lower() not in header_map]
                if missing_cols:
                    raise ValueError(
                        f"❌ 【表头错误】dataset.csv必须包含{', '.join(required_cols)}列\n"
                        f"📋 当前表头：{reader.fieldnames}"
                    )
                
                # 检查是否有VALUE列（可选）
                has_value_col = "value" in header_map
                dataset_col = header_map["dataset"]
                path_col = header_map["path"]
                value_col = header_map["value"] if has_value_col else None
                
                for line_num, row in enumerate(reader, 2):
                    dataset_name = row[dataset_col].strip().lower()
                    path_value = row[path_col].strip()
                    
                    if not dataset_name:
                        print(f"⚠️  [WARNING] dataset.csv第{line_num}行DATASET列为空，跳过")
                        continue
                    if not path_value:
                        print(f"⚠️  [WARNING] dataset.csv第{line_num}行PATH为空，{dataset_name} → 跳过")
                        continue
                    
                    # 解析VALUE列（如果存在且非空）
                    value_data = None
                    if has_value_col and value_col:
                        # 使用 get 方法，提供默认值
                        raw_value = row.get(value_col, "")
                        if raw_value is not None:
                            value_str = str(raw_value).strip()
                            if value_str and value_str.lower() != "null":
                                try:
                                    # 尝试解析JSON
                                    value_data = json.loads(value_str)
                                    print(f"📝 【dataset.csv行{line_num}】{dataset_name} → VALUE解析成功: {value_data}")
                                except json.JSONDecodeError as e:
                                    print(f"⚠️  [WARNING] dataset.csv第{line_num}行VALUE列JSON格式错误: {e}，内容: '{value_str}'")
                                    value_data = None
                    
                    dataset_map[dataset_name] = {
                        "path": path_value,
                        "value": value_data  # 可能是None、字典、列表或其他JSON数据
                    }
                    
                    value_log = f"额外配置: {value_data}" if value_data is not None else "无额外配置"
                    print(f"📝 【dataset.csv行{line_num}】{dataset_name} → PATH={path_value}, {value_log}")
            
            print(f"\n✅ 【dataset.csv映射】共加载{len(dataset_map)}条数据")
            return dataset_map
        except Exception as e:
            raise Exception(f"❌  读取dataset.csv失败：{str(e)}") from e