"""
基础损失函数类

为自定义损失函数提供统一的接口和基础功能
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseCustomLoss(ABC):
    """
    自定义损失函数基类
    
    提供自定义损失函数的基础接口和通用功能
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化基础损失函数
        
        Args:
            name (str): 损失函数名称
            description (str): 损失函数描述
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def create_loss(self, config: Dict[str, Any]) -> nn.Module:
        """
        创建损失函数实例
        
        Args:
            config (dict): 损失函数配置参数
            
        Returns:
            nn.Module: 损失函数实例
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
            "reduction": "mean"
        }