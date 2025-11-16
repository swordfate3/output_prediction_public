# -*- coding: utf-8 -*-
"""
models 模型包初始化文件

这个包包含了项目的机器学习模型实现，包括：
- lstm_model: LSTM神经网络模型实现
- mamba_model: Mamba状态空间模型实现
- trainer: 模型训练器和评估工具

Author: Output Prediction Project
"""

# 导入主要的模型类和训练器，方便外部使用
try:
    from .lstm_model import CipherLSTM
    from .mamba_model import CipherMamba, MambaBlock, SelectiveScan
    # [ADD] 新增: 导入统一接口的 iTransformer 模型
    from .iTransformer import CipherITransformer
    
except ImportError as e:
    # 如果某些模块导入失败，记录但不中断整个包的加载
    import warnings
    warnings.warn(f"部分模型模块导入失败: {e}", ImportWarning)

__version__ = "1.0.0"
__author__ = "Output Prediction Project"
__all__ = [
    "CipherLSTM",
    "CipherMamba",
    "MambaBlock",
    "SelectiveScan",
    # [ADD] 新增: 将 CipherITransformer 加入导出列表
    "CipherITransformer",
]