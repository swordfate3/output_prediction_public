#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录管理器使用示例

展示如何使用统一的目录管理工具来管理项目中的各种目录。

Author: Assistant
Date: 2025-01-21
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.directory_manager import (
    DirectoryManager,
    get_global_directory_manager,
    get_logs_directory,
    get_plots_directory,
    get_data_directory,
    ensure_directory
)
import logging


def demonstrate_basic_usage():
    """
    演示基本的目录管理功能
    """
    print("=== 基本目录管理功能演示 ===")
    
    # 创建目录管理器实例
    dm = DirectoryManager()
    
    # 获取各种类型的目录
    logs_dir = dm.get_directory('logs')
    plots_dir = dm.get_directory('plots')
    data_dir = dm.get_directory('data')
    
    print(f"日志目录: {logs_dir}")
    print(f"图表目录: {plots_dir}")
    print(f"数据目录: {data_dir}")
    
    # 创建子目录
    experiment_dir = dm.create_subdirectory('data', 'experiment_001')
    print(f"实验目录: {experiment_dir}")
    
    # 创建带时间戳的目录
    timestamped_dir = dm.create_timestamped_directory('results', 'training')
    print(f"时间戳目录: {timestamped_dir}")
    
    # 获取文件路径
    log_file = dm.get_file_path('logs', 'application.log')
    plot_file = dm.get_file_path('plots', 'training_curves.png')
    
    print(f"日志文件路径: {log_file}")
    print(f"图表文件路径: {plot_file}")


def demonstrate_global_manager():
    """
    演示全局目录管理器的使用
    """
    print("\n=== 全局目录管理器演示 ===")
    
    # 使用全局目录管理器
    global_dm = get_global_directory_manager()
    
    # 列出所有配置的目录
    directories = global_dm.list_directories()
    print("已配置的目录:")
    for dir_type, path in directories.items():
        print(f"  {dir_type}: {path}")


def demonstrate_convenience_functions():
    """
    演示便捷函数的使用
    """
    print("\n=== 便捷函数演示 ===")
    
    # 使用便捷函数获取目录
    logs_dir = get_logs_directory()
    plots_dir = get_plots_directory()
    data_dir = get_data_directory()
    
    print(f"日志目录 (便捷函数): {logs_dir}")
    print(f"图表目录 (便捷函数): {plots_dir}")
    print(f"数据目录 (便捷函数): {data_dir}")
    
    # 确保自定义目录存在
    custom_dir = ensure_directory('./custom_output')
    print(f"自定义目录: {custom_dir}")


def demonstrate_logger_integration():
    """
    演示与日志系统的集成
    """
    print("\n=== 日志系统集成演示 ===")
    
    # 使用标准日志器
    logger = logging.getLogger(__name__)
    logger.info("使用标准日志器的日志消息")
    
    # 配置日志输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("日志系统已简化为标准logging模块")
    print("日志系统已简化，使用Python标准logging模块")


def demonstrate_configuration():
    """
    演示目录配置的保存和加载
    """
    print("\n=== 配置管理演示 ===")
    
    # 创建目录管理器并自定义配置
    dm = DirectoryManager()
    dm.set_directory('custom_type', './custom_directory')
    dm.set_directory('temp', './temporary_files')
    
    # 保存配置
    config_file = './config/directory_config.json'
    try:
        dm.save_config(config_file)
        print(f"配置已保存到: {config_file}")
        
        # 加载配置
        new_dm = DirectoryManager(config_file=config_file)
        print("配置加载成功")
        
        # 显示加载的配置
        loaded_dirs = new_dm.list_directories()
        print("加载的目录配置:")
        for dir_type, path in loaded_dirs.items():
            print(f"  {dir_type}: {path}")
            
    except Exception as e:
        print(f"配置操作失败: {e}")


if __name__ == "__main__":
    print("目录管理器使用示例")
    print("=" * 50)
    
    try:
        demonstrate_basic_usage()
        demonstrate_global_manager()
        demonstrate_convenience_functions()
        demonstrate_logger_integration()
        demonstrate_configuration()
        
        print("\n=== 演示完成 ===")
        print("所有功能演示成功完成！")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()