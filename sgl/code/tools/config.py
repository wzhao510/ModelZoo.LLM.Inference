#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - 配置管理模块
该模块负责定义和管理程序的所有配置项，包括路径设置、默认值等。
"""

import os


class PathConfig:
    """
    路径配置类
    定义程序中使用的所有路径常量
    """
    def __init__(self):
        self.TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CSV_ROOT_DIR = os.path.join(self.TOOLS_DIR, "summary")
        self.START_SCRIPT = "start_docker.py"
        self.DOCKER_VOLUMES = ["/external/:/external/", "/mnt/:/mnt/", "/mxstorage/:/mxstorage/"]
        self.TARGET_PATH = "../.."
        self.OUTPUT_PATH = "../../outputs"
        self.MACHINE_CONFIG = "../../models/machines.json"
        self.DAILY_RESULT_DIR = "../../daily_result"
        self.MODELS_ROOT_DIR = "../../models/"


class ValidationConfig:
    """
    验证配置类
    定义配置验证所需的字段和规则
    """
    SETTING_REQUIRED_FIELDS = [
        "DATE", "CONTAINER_IMAGE", "LOCAL_IP", "RUN_USER",
        "INCREMENTAL_MODE", "PARALLEL", "RM_EXIST_DOCKER", "PULL_IMAGES",
        "ALL_RUN", "CONFIG_REUSE"
    ]
    SETTING_BOOL_FIELDS = [
        "INCREMENTAL_MODE", "PARALLEL", "RM_EXIST_DOCKER", "PULL_IMAGES", "CONFIG_REUSE"
    ]
    DATASET_PARAM_MAP = {
        "random": "--dataset-path",
        "ceval": "--test_jsonl",
        "mmlu": "--data_dir"
    }
    MODEL_PATH_PATTERN = r'(--model-path)\s+/models/[^"\s]+'
    SPECULATIVE_DRAFT_MODEL_PATTERN = r'(--speculative-draft-model-path)\s+/models/[^"\s]+'
    CEVAL_MODEL_PATTERN = r'(--model)\s+/models/[^"\s]+'
    ILLEGAL_SEPARATORS = [",", " ", "，", "|", "；"]
    VALID_BENCHMARKS = {"random", "ceval", "mmlu"}


class Constants:
    """
    常量定义类
    定义程序中使用的各种常量
    """
    pass