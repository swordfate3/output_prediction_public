# -*- coding: utf-8 -*-
"""
experiments 实验包初始化文件

这个包包含了项目的各种实验脚本，包括：
- exp1_hyperopt: 超参数优化实验
- exp2_attack: 攻击实验
- exp3_variants: 变种实验

Author: Output Prediction Project
"""

# 实验模块通常不需要导入，因为它们是独立的脚本
# 但可以在这里定义一些共用的实验配置或工具函数

__version__ = "1.0.0"
__author__ = "Output Prediction Project"
__all__ = ["exp1_hyperopt", "exp2_attack", "exp3_variants"]

# 实验相关的常量定义
# EXPERIMENT_BASE_DIR = "../results"
# DEFAULT_EXPERIMENT_CONFIG = {
#     "batch_size": 32,
#     "epochs": 100,
#     "learning_rate": 0.001
# }