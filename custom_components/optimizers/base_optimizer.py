"""
基础优化器类

为自定义优化器提供统一的接口和基础功能
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseCustomOptimizer(ABC):
    """
    自定义优化器基类
    
    提供自定义优化器的基础接口和通用功能
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化基础优化器
        
        Args:
            name (str): 优化器名称
            description (str): 优化器描述
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def create_optimizer(self, model_parameters,
                         config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        创建优化器实例
        
        Args:
            model_parameters: 模型参数
            config (dict): 优化器配置参数
            
        Returns:
            torch.optim.Optimizer: 优化器实例
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置参数
        
        Args:
            config (dict): 配置参数
            
        Returns:
            bool: 配置是否有效
        """
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            dict: 默认配置参数
        """
        return {
            "learning_rate": 0.001
        }