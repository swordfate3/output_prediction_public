#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录管理工具模块

提供统一的目录管理功能，包括目录创建、路径规范化、配置管理等。
用于替代项目中分散的目录创建操作，提供灵活的目录配置管理。

Author: Assistant
Date: 2025-01-21
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class DirectoryManager:
    """
    目录管理器

    提供统一的目录创建、配置和管理功能。
    支持预定义目录类型和自定义目录配置。
    """

    # 预定义的目录类型配置
    DEFAULT_DIRECTORIES = {
        "logs": "./logs",
        "plots": "./plots",
        "data": "./data",
        "models": "./models",
        "results": "./results",
        "temp": "./temp",
        "cache": "./cache",
    }

    def __init__(self, base_dir: str = ".", config_file: Optional[str] = None):
        """
        初始化目录管理器

        Args:
            base_dir (str): 基础目录路径，默认为当前目录
            config_file (str, optional): 目录配置文件路径
        """
        self.base_dir = Path(base_dir).resolve()
        self.config_file = config_file
        self.directories = self.DEFAULT_DIRECTORIES.copy()

        # 如果提供了配置文件，加载配置
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def set_directory(self, dir_type: str, path: str) -> None:
        """
        设置目录类型对应的路径

        Args:
            dir_type (str): 目录类型名称
            path (str): 目录路径
        """
        self.directories[dir_type] = path

    def get_directory(self, dir_type: str, create: bool = True) -> str:
        """
        获取指定类型的目录路径

        Args:
            dir_type (str): 目录类型名称
            create (bool): 是否自动创建目录，默认为True

        Returns:
            str: 目录的绝对路径

        Raises:
            ValueError: 当目录类型不存在时抛出异常
        """
        if dir_type not in self.directories:
            raise ValueError(f"未知的目录类型: {dir_type}")

        # 获取相对路径并转换为绝对路径
        relative_path = self.directories[dir_type]
        if os.path.isabs(relative_path):
            abs_path = relative_path
        else:
            abs_path = str(self.base_dir / relative_path)

        abs_path = str(Path(abs_path).resolve())

        # 如果需要，创建目录
        if create:
            Path(abs_path).mkdir(parents=True, exist_ok=True)

        return str(abs_path)

    def get_file_path(
        self, dir_type: str, filename: str, create_dir: bool = True
    ) -> str:
        """
        获取指定目录类型下的文件完整路径

        Args:
            dir_type (str): 目录类型名称
            filename (str): 文件名
            create_dir (bool): 是否自动创建目录，默认为True

        Returns:
            str: 文件的完整路径
        """
        dir_path = self.get_directory(dir_type, create=create_dir)
        return os.path.join(dir_path, filename)

    def create_subdirectory(self, dir_type: str, subdir: str) -> str:
        """
        在指定目录类型下创建子目录

        Args:
            dir_type (str): 父目录类型名称
            subdir (str): 子目录名称

        Returns:
            str: 子目录的完整路径
        """
        parent_dir = self.get_directory(dir_type, create=True)
        subdir_path = os.path.join(parent_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path

    def create_timestamped_directory(
        self, dir_type: str, prefix: str = ""
    ) -> str:
        """
        在指定目录类型下创建带时间戳的子目录

        Args:
            dir_type (str): 父目录类型名称
            prefix (str): 目录名前缀

        Returns:
            str: 带时间戳目录的完整路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir_name = f"{prefix}_{timestamp}" if prefix else timestamp
        return self.create_subdirectory(dir_type, subdir_name)

    def list_directories(self) -> Dict[str, str]:
        """
        列出所有已配置的目录类型和路径

        Returns:
            Dict[str, str]: 目录类型到路径的映射
        """
        result = {}
        for dir_type, relative_path in self.directories.items():
            try:
                result[dir_type] = self.get_directory(dir_type, create=False)
            except Exception:
                result[dir_type] = relative_path
        return result

    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        保存当前目录配置到文件

        Args:
            config_file (str, optional): 配置文件路径，默认使用初始化时的配置文件
        """
        config_path = config_file or self.config_file
        if not config_path:
            raise ValueError("未指定配置文件路径")

        config_data = {
            "base_dir": str(self.base_dir),
            "directories": self.directories,
            "created_at": datetime.now().isoformat(),
        }

        # 确保配置文件目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    def load_config(self, config_file: str) -> None:
        """
        从文件加载目录配置

        Args:
            config_file (str): 配置文件路径
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if "directories" in config_data:
            self.directories.update(config_data["directories"])

        if "base_dir" in config_data:
            self.base_dir = Path(config_data["base_dir"]).resolve()

    def cleanup_empty_directories(
        self, dir_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        清理空目录

        Args:
            dir_types (List[str], optional): 要清理的目录类型列表，默认清理所有

        Returns:
            List[str]: 被删除的目录路径列表
        """
        removed_dirs = []
        target_types = dir_types or list(self.directories.keys())

        for dir_type in target_types:
            try:
                dir_path = self.get_directory(dir_type, create=False)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    removed_dirs.append(dir_path)
            except Exception:
                continue

        return removed_dirs


# 全局目录管理器实例
_global_directory_manager = None


def get_global_directory_manager(
    base_dir: str = ".", config_file: Optional[str] = None
) -> DirectoryManager:
    """
    获取全局目录管理器实例

    Args:
        base_dir (str): 基础目录路径
        config_file (str, optional): 配置文件路径

    Returns:
        DirectoryManager: 全局目录管理器实例
    """
    global _global_directory_manager
    if _global_directory_manager is None:
        _global_directory_manager = DirectoryManager(base_dir, config_file)
    return _global_directory_manager


def reset_global_directory_manager() -> None:
    """
    重置全局目录管理器实例
    """
    global _global_directory_manager
    _global_directory_manager = None


# 便捷函数
def ensure_directory(path: str) -> str:
    """
    确保目录存在，如果不存在则创建

    Args:
        path (str): 目录路径

    Returns:
        str: 目录的绝对路径
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_logs_directory(create: bool = True) -> str:
    """
    获取日志目录路径

    Args:
        create (bool): 是否自动创建目录

    Returns:
        str: 日志目录路径
    """
    return get_global_directory_manager().get_directory("logs", create=create)


def get_plots_directory(create: bool = True) -> str:
    """
    获取图表目录路径

    Args:
        create (bool): 是否自动创建目录

    Returns:
        str: 图表目录路径
    """
    return get_global_directory_manager().get_directory("plots", create=create)


def get_data_directory(create: bool = True) -> str:
    """
    获取数据目录路径

    Args:
        create (bool): 是否自动创建目录

    Returns:
        str: 数据目录路径
    """
    return get_global_directory_manager().get_directory("data", create=create)


def get_models_directory(create: bool = True) -> str:
    """
    获取模型目录路径

    Args:
        create (bool): 是否自动创建目录

    Returns:
        str: 模型目录路径
    """
    return get_global_directory_manager().get_directory(
        "models", create=create
    )


def get_results_directory(create: bool = True) -> str:
    """
    获取结果目录路径

    Args:
        create (bool): 是否自动创建目录

    Returns:
        str: 结果目录路径
    """
    return get_global_directory_manager().get_directory(
        "results", create=create
    )
