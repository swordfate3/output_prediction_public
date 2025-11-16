# -*- coding: utf-8 -*-
"""
utils 工具包初始化文件

这个包包含了项目的各种工具模块，包括：
- config: 配置管理工具
- data_generator: 数据生成工具
- metrics: 评估指标计算工具
- logger: 统一日志记录工具
- directory_manager: 目录管理工具

Author: Output Prediction Project
"""

# 导入主要的工具类和函数，方便外部使用
try:
    from .config import *
    from .data_generator import *
    from .metrics import *
    from .logger import *
    from .directory_manager import *
except ImportError as e:
    # 如果某些模块导入失败，记录但不中断整个包的加载
    import warnings
    warnings.warn(f"部分工具模块导入失败: {e}", ImportWarning)

__version__ = "1.0.0"
__author__ = "Output Prediction Project"
__all__ = [
    # config 模块
    "Config",
    "config",
    
    # data_generator 模块
    "extractTargetBits",
    "generate_dataset",
    
    # metrics 模块
    "bitwise_success_rate",
    "log2_success_rate",
    
    # logger 模块
    "Logger",
    "getGlobalLogger",
    "resetGlobalLogger",
    
    # directory_manager 模块
    "DirectoryManager",
    "get_global_directory_manager",
    "reset_global_directory_manager",
    "ensure_directory",
    "get_logs_directory",
    "get_plots_directory",
    "get_data_directory",
    "get_models_directory",
    "get_results_directory"
]