"""
自定义损失函数实现

提供各种自定义损失函数的具体实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_loss import BaseCustomLoss


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    用于解决类别不平衡问题的损失函数
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha (float): 平衡因子
            gamma (float): 聚焦参数
            reduction (str): 损失计算方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            inputs (torch.Tensor): 预测值
            targets (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 损失值
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
                                                     reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    加权二元交叉熵损失
    
    为正负样本分配不同权重的BCE损失函数
    """
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        """
        初始化加权BCE损失
        
        Args:
            pos_weight (float): 正样本权重
            reduction (str): 损失计算方式
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            inputs (torch.Tensor): 预测值
            targets (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 损失值
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor(self.pos_weight),
            reduction=self.reduction
        )


class FocalLossCustom(BaseCustomLoss):
    """
    自定义Focal Loss
    
    基于Focal Loss的自定义损失函数实现
    """
    
    def __init__(self):
        """
        初始化Focal Loss自定义损失函数
        """
        super().__init__(
            name="focal_loss",
            description="Focal Loss，用于解决类别不平衡问题"
        )
    
    def create_loss(self, config: Dict[str, Any]) -> nn.Module:
        """
        创建Focal Loss实例
        
        Args:
            config (dict): 损失函数配置参数
            
        Returns:
            nn.Module: Focal Loss实例
        """
        alpha = config.get("alpha", 1.0)
        gamma = config.get("gamma", 2.0)
        reduction = config.get("reduction", "mean")
        
        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            dict: 默认配置参数
        """
        return {
            "alpha": 1.0,
            "gamma": 2.0,
            "reduction": "mean"
        }


class WeightedBCELossCustom(BaseCustomLoss):
    """
    自定义加权BCE损失
    
    基于加权BCE的自定义损失函数实现
    """
    
    def __init__(self):
        """
        初始化加权BCE自定义损失函数
        """
        super().__init__(
            name="weighted_bce",
            description="加权二元交叉熵损失，支持正负样本权重调节"
        )
    
    def create_loss(self, config: Dict[str, Any]) -> nn.Module:
        """
        创建加权BCE损失实例
        
        Args:
            config (dict): 损失函数配置参数
            
        Returns:
            nn.Module: 加权BCE损失实例
        """
        pos_weight = config.get("pos_weight", 1.0)
        reduction = config.get("reduction", "mean")
        
        return WeightedBCELoss(pos_weight=pos_weight, reduction=reduction)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            dict: 默认配置参数
        """
        return {
            "pos_weight": 1.0,
            "reduction": "mean"
        }


# 损失函数注册表
CUSTOM_LOSSES = {
    "focal_loss": FocalLossCustom(),
    "weighted_bce": WeightedBCELossCustom(),
}


def get_custom_loss(loss_name: str) -> BaseCustomLoss:
    """
    获取自定义损失函数
    
    根据名称获取对应的自定义损失函数实例
    
    Args:
        loss_name (str): 损失函数名称
        
    Returns:
        BaseCustomLoss: 自定义损失函数实例
        
    Raises:
        ValueError: 当损失函数名称不存在时
        
    Example:
        >>> loss_fn = get_custom_loss("focal_loss")
        >>> config = {"alpha": 1.0, "gamma": 2.0}
        >>> criterion = loss_fn.create_loss(config)
    """
    if loss_name not in CUSTOM_LOSSES:
        available = list(CUSTOM_LOSSES.keys())
        raise ValueError(f"未知的自定义损失函数: {loss_name}. "
                        f"可用的损失函数: {available}")
    
    return CUSTOM_LOSSES[loss_name]


def list_custom_losses() -> Dict[str, str]:
    """
    列出所有可用的自定义损失函数
    
    Returns:
        dict: 损失函数名称和描述的字典
        
    Example:
        >>> losses = list_custom_losses()
        >>> for name, desc in losses.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: loss.description
            for name, loss in CUSTOM_LOSSES.items()}