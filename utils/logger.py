#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的日志系统模块

使用Python标准logging模块提供基础的日志记录功能。
去除了复杂的自定义封装，直接使用标准logger。

Author: Assistant
Date: 2025-01-05
"""

import logging
import os
from .directory_manager import get_logs_directory

# 全局logger实例
_logger = None


def getGlobalLogger(name: str = "application") -> logging.Logger:
    """
    获取全局日志器实例
    
    使用Python标准logging模块创建简单的日志器。
    
    Args:
        name (str): 日志器名称，默认为"application"
        
    Returns:
        logging.Logger: 标准日志器实例
        
    Example:
        >>> logger = getGlobalLogger()
        >>> logger.info("应用启动")
    """
    global _logger
    if _logger is None:
        _logger = _setupLogger(name)
    return _logger


def _setupLogger(name: str) -> logging.Logger:
    """
    设置标准日志器
    
    创建基础的日志器配置，包括文件和控制台输出。
    
    Args:
        name (str): 日志器名称
        
    Returns:
        logging.Logger: 配置好的标准日志器
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 获取日志目录
    log_dir = get_logs_directory(create=True)
    log_file = os.path.join(log_dir, "application.log")
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 简单的日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def resetGlobalLogger():
    """
    重置全局日志器
    
    清除当前的全局日志器实例。
    """
    global _logger
    _logger = None
