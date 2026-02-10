#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_operations.py - 文件操作模块
该模块提供通用的文件和目录操作功能，如复制、删除、路径处理等。
"""

import os
import shutil
import re


class FileOperations:
    """
    文件操作工具类：提供通用的文件和目录操作
    """
    def __init__(self, path_manager):
        self.pm = path_manager

    def get_abs_path(self, relative_path: str) -> str:
        """
        将相对路径转换为绝对路径
        :param relative_path: 相对路径
        :return: 绝对路径
        """
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.abspath(os.path.join(self.pm.TOOLS_DIR, relative_path))

    def clear_directory(self, dir_path: str) -> None:
        """
        清空目录内容（保留目录本身）
        :param dir_path: 目标目录路径
        """
        if not os.path.exists(dir_path):
            print(f"ℹ️  [信息] 目录{dir_path}不存在，无需清空")
            return
        
        try:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    print(f"🗑️  已删除文件：{item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"🗑️  已删除目录：{item_path}")
            print(f"✅  成功清空目录：{dir_path}")
        except Exception as e:
            print(f"❌  清空目录{dir_path}失败：{str(e)}")
            raise

    def copy_file(self, src: str, dst: str) -> None:
        """
        复制文件
        :param src: 源文件路径
        :param dst: 目标文件路径
        """
        try:
            shutil.copy2(src, dst)
            print(f"✅  复制成功：{os.path.basename(src)} → {dst}")
        except Exception as e:
            print(f"❌  复制失败：{src} → {dst}，错误：{str(e)}")
            raise

    def get_json_block(self, content: str, block_key: str) -> tuple[str, int, int]:
        """
        从JSON字符串中定位指定块（如"server_cmds": { ... }）
        :param content: JSON字符串
        :param block_key: 块名称（如"server_cmds"）
        :return: (块内容, 块起始位置, 块结束位置)，未找到返回("", -1, -1)
        """
        block_start_str = f'"{block_key}": {{'
        start_idx = content.find(block_start_str)
        if start_idx == -1:
            return "", -1, -1
        
        # 统计大括号层数，找到块结束位置
        current_pos = start_idx + len(block_start_str)
        brace_count = 1
        end_idx = -1
        while current_pos < len(content) and brace_count > 0:
            char = content[current_pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = current_pos
                    break
            current_pos += 1
        
        if end_idx == -1:
            return "", -1, -1
        return content[start_idx:end_idx+1], start_idx, end_idx

    def get_json_sub_block(self, parent_block: str, sub_block_key: str) -> tuple[str, int, int]:
        """
        从父JSON块中定位子块（如benchmark_cmds中的ceval子块）
        :param parent_block: 父块内容
        :param sub_block_key: 子块名称（如"ceval"）
        :return: (子块内容, 子块起始位置, 子块结束位置)，未找到返回("", -1, -1)
        """
        return self.get_json_block(parent_block, sub_block_key)