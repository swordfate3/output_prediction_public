"""
自定义损失函数模块

提供各种自定义损失函数的实现和工厂函数
"""

from .custom_losses import get_custom_loss, list_custom_losses

__all__ = ['get_custom_loss', 'list_custom_losses']