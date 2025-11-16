"""
自定义组件包

提供自定义优化器和损失函数的实现，方便扩展和维护
"""

try:
    from .optimizers import get_custom_optimizer
    from .losses import get_custom_loss
    
    __all__ = [
        'get_custom_optimizer',
        'get_custom_loss'
    ]
except ImportError:
    # 如果导入失败，提供空的实现
    __all__ = []