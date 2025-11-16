import sys
import os
from models.train_model import TrainModel
from custom_components.optimizers import list_custom_optimizers
from custom_components.losses import list_custom_losses
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
自定义组件使用示例

演示如何使用自定义优化器和损失函数进行训练
"""


def listAvailableCustomComponents():
    """
    列出所有可用的自定义组件
    
    显示当前可用的自定义优化器和损失函数
    """
    print("=== 可用的自定义优化器 ===")
    optimizers = list_custom_optimizers()
    for name, description in optimizers.items():
        print(f"- {name}: {description}")
    
    print("\n=== 可用的自定义损失函数 ===")
    losses = list_custom_losses()
    for name, description in losses.items():
        print(f"- {name}: {description}")


def trainWithCustomAdamW():
    """
    使用自定义AdamW优化器进行训练
    
    演示如何配置和使用自定义AdamW优化器
    """
    print("\n=== 使用自定义AdamW优化器训练 ===")
    
    # 创建训练器实例
    trainer = TrainModel()
    
    # 设置训练参数，使用自定义AdamW优化器
    trainer.train_parameters = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "optimizer": "adamw_custom",  # 使用自定义AdamW
        "lr": 0.001,
        "weight_decay": 0.01,  # AdamW权重衰减
        "betas": (0.9, 0.999),  # AdamW beta参数
        "criterion": {"type": "bce", "reduction": "mean"}
    }
    
    print("配置参数:")
    print(f"- 优化器: {trainer.train_parameters['optimizer']}")
    print(f"- 学习率: {trainer.train_parameters['lr']}")
    print(f"- 权重衰减: {trainer.train_parameters['weight_decay']}")
    print(f"- Beta参数: {trainer.train_parameters['betas']}")
    
    # 注意：这里只是演示配置，实际训练需要提供数据
    print("训练配置完成！")


def trainWithFocalLoss():
    """
    使用自定义Focal Loss进行训练
    
    演示如何配置和使用自定义Focal Loss损失函数
    """
    print("\n=== 使用自定义Focal Loss训练 ===")
    
    # 创建训练器实例
    trainer = TrainModel()
    
    # 设置训练参数，使用自定义Focal Loss
    trainer.train_parameters = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "optimizer": "adam",
        "lr": 0.001,
        "criterion": {
            "type": "focal_loss",  # 使用自定义Focal Loss
            "alpha": 1.0,  # 平衡因子
            "gamma": 2.0,  # 聚焦参数
            "reduction": "mean"
        }
    }
    
    print("配置参数:")
    print(f"- 损失函数: {trainer.train_parameters['criterion']['type']}")
    print(f"- Alpha参数: {trainer.train_parameters['criterion']['alpha']}")
    print(f"- Gamma参数: {trainer.train_parameters['criterion']['gamma']}")
    
    # 注意：这里只是演示配置，实际训练需要提供数据
    print("训练配置完成！")


def trainWithBothCustomComponents():
    """
    同时使用自定义优化器和损失函数
    
    演示如何同时配置自定义优化器和损失函数
    """
    print("\n=== 同时使用自定义优化器和损失函数 ===")
    
    # 创建训练器实例
    trainer = TrainModel()
    
    # 设置训练参数，同时使用自定义组件
    trainer.train_parameters = {
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "optimizer": "sgd_momentum_custom",  # 自定义SGD动量优化器
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "nesterov": True,
        "criterion": {
            "type": "weighted_bce",  # 自定义加权BCE损失
            "pos_weight": 2.0,  # 正样本权重
            "reduction": "mean"
        }
    }
    
    print("配置参数:")
    print(f"- 优化器: {trainer.train_parameters['optimizer']}")
    print(f"- 学习率: {trainer.train_parameters['lr']}")
    print(f"- 动量: {trainer.train_parameters['momentum']}")
    print(f"- Nesterov: {trainer.train_parameters['nesterov']}")
    print(f"- 损失函数: {trainer.train_parameters['criterion']['type']}")
    print(f"- 正样本权重: {trainer.train_parameters['criterion']['pos_weight']}")
    
    # 注意：这里只是演示配置，实际训练需要提供数据
    print("训练配置完成！")


def demonstrateCustomComponentsUsage():
    """
    完整的自定义组件使用演示
    
    展示如何在实际项目中使用自定义组件
    """
    print("自定义组件使用演示")
    print("=" * 50)
    
    # 列出可用组件
    listAvailableCustomComponents()
    
    # 演示不同的配置方式
    trainWithCustomAdamW()
    trainWithFocalLoss()
    trainWithBothCustomComponents()
    
    print("\n=== 使用说明 ===")
    print("1. 自定义优化器支持:")
    print("   - adamw_custom: 自定义AdamW优化器")
    print("   - sgd_momentum_custom: 自定义SGD动量优化器")
    print()
    print("2. 自定义损失函数支持:")
    print("   - focal_loss: Focal Loss，用于类别不平衡")
    print("   - weighted_bce: 加权BCE损失")
    print()
    print("3. 配置方式:")
    print("   - 在train_parameters中设置optimizer和criterion")
    print("   - 自定义组件会自动被识别和使用")
    print("   - 如果组件不存在，会回退到内置组件")


if __name__ == "__main__":
    demonstrateCustomComponentsUsage()