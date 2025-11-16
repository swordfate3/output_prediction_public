"""
自定义优化器实现

提供各种自定义优化器的具体实现
"""

import torch
import torch.optim as optim
from typing import Dict, Any
from .base_optimizer import BaseCustomOptimizer


class AdamWCustomOptimizer(BaseCustomOptimizer):
    """
    自定义AdamW优化器
    
    基于PyTorch的AdamW优化器，添加了自定义配置选项
    """
    
    def __init__(self):
        """
        初始化AdamW自定义优化器
        """
        super().__init__(
            name="adamw_custom",
            description="自定义AdamW优化器，支持权重衰减和学习率调度"
        )
    
    def create_optimizer(self, model_parameters,
                         config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        创建AdamW优化器实例
        
        Args:
            model_parameters: 模型参数
            config (dict): 优化器配置参数
            
        Returns:
            torch.optim.AdamW: AdamW优化器实例
        """
        lr = config.get("learning_rate", 0.001)
        weight_decay = config.get("weight_decay", 0.01)
        betas = config.get("betas", (0.9, 0.999))
        eps = config.get("eps", 1e-8)
        
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            dict: 默认配置参数
        """
        return {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }


class SGDMomentumCustomOptimizer(BaseCustomOptimizer):
    """
    自定义SGD动量优化器
    
    基于PyTorch的SGD优化器，添加了动量和权重衰减支持
    """
    
    def __init__(self):
        """
        初始化SGD动量自定义优化器
        """
        super().__init__(
            name="sgd_momentum_custom",
            description="自定义SGD动量优化器，支持动量和权重衰减"
        )
    
    def create_optimizer(self, model_parameters,
                         config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        创建SGD优化器实例
        
        Args:
            model_parameters: 模型参数
            config (dict): 优化器配置参数
            
        Returns:
            torch.optim.SGD: SGD优化器实例
        """
        lr = config.get("learning_rate", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 0.0001)
        nesterov = config.get("nesterov", False)
        
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            dict: 默认配置参数
        """
        return {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "nesterov": False
        }


# 优化器注册表
CUSTOM_OPTIMIZERS = {
    "adamw_custom": AdamWCustomOptimizer(),
    "sgd_momentum_custom": SGDMomentumCustomOptimizer(),
}


def get_custom_optimizer(optimizer_name: str) -> BaseCustomOptimizer:
    """
    获取自定义优化器
    
    根据名称获取对应的自定义优化器实例
    
    Args:
        optimizer_name (str): 优化器名称
        
    Returns:
        BaseCustomOptimizer: 自定义优化器实例
        
    Raises:
        ValueError: 当优化器名称不存在时
        
    Example:
        >>> optimizer = get_custom_optimizer("adamw_custom")
        >>> config = {"learning_rate": 0.001, "weight_decay": 0.01}
        >>> opt_instance = optimizer.create_optimizer(
        ...     model.parameters(), config)
    """
    if optimizer_name not in CUSTOM_OPTIMIZERS:
        available = list(CUSTOM_OPTIMIZERS.keys())
        raise ValueError(f"未知的自定义优化器: {optimizer_name}. "
                         f"可用的优化器: {available}")
    
    return CUSTOM_OPTIMIZERS[optimizer_name]


def list_custom_optimizers() -> Dict[str, str]:
    """
    列出所有可用的自定义优化器
    
    Returns:
        dict: 优化器名称和描述的字典
        
    Example:
        >>> optimizers = list_custom_optimizers()
        >>> for name, desc in optimizers.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: opt.description 
            for name, opt in CUSTOM_OPTIMIZERS.items()}