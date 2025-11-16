# 自定义组件使用指南

本目录包含了项目的自定义优化器和损失函数实现，提供了一个简单而灵活的方式来扩展训练功能。

## 目录结构

```
custom_components/
├── __init__.py              # 主模块初始化
├── README.md               # 使用说明（本文件）
├── optimizers/             # 自定义优化器模块
│   ├── __init__.py         # 优化器模块初始化
│   ├── base_optimizer.py   # 优化器基类
│   └── custom_optimizers.py # 具体优化器实现
└── losses/                 # 自定义损失函数模块
    ├── __init__.py         # 损失函数模块初始化
    ├── base_loss.py        # 损失函数基类
    └── custom_losses.py    # 具体损失函数实现
```

## 可用的自定义组件

### 自定义优化器

1. **adamw_custom** - 自定义AdamW优化器
   - 支持权重衰减和自适应学习率
   - 适用于大多数深度学习任务

2. **sgd_momentum_custom** - 自定义SGD动量优化器
   - 支持动量和Nesterov加速
   - 适用于需要稳定训练的场景

### 自定义损失函数

1. **focal_loss** - Focal Loss
   - 用于处理类别不平衡问题
   - 可调节alpha和gamma参数

2. **weighted_bce** - 加权二元交叉熵损失
   - 支持正样本权重调节
   - 适用于不平衡二分类任务

## 使用方法

### 1. 基本配置

在训练参数中指定自定义组件：

```python
from models.train_model import TrainModel

trainer = TrainModel()
trainer.train_parameters = {
    "optimizer": "adamw_custom",  # 使用自定义AdamW
    "lr": 0.001,
    "weight_decay": 0.01,
    "criterion": {
        "type": "focal_loss",  # 使用自定义Focal Loss
        "alpha": 1.0,
        "gamma": 2.0
    }
}
```

### 2. 查看可用组件

```python
from custom_components.optimizers import list_custom_optimizers
from custom_components.losses import list_custom_losses

# 列出所有可用的自定义优化器
optimizers = list_custom_optimizers()
print(optimizers)

# 列出所有可用的自定义损失函数
losses = list_custom_losses()
print(losses)
```

### 3. 运行示例

```bash
cd examples
python custom_components_example.py
```

## 添加新的自定义组件

### 添加新的优化器

1. 在 `custom_components/optimizers/custom_optimizers.py` 中创建新类：

```python
class YourCustomOptimizer(BaseCustomOptimizer):
    def create_optimizer(self, model_parameters, **kwargs):
        # 实现你的优化器逻辑
        return torch.optim.YourOptimizer(model_parameters, **kwargs)
    
    def get_default_config(self):
        return {
            "lr": 0.001,
            # 其他默认参数
        }
```

2. 在 `CUSTOM_OPTIMIZERS` 字典中注册：

```python
CUSTOM_OPTIMIZERS = {
    # 现有优化器...
    "your_optimizer": YourCustomOptimizer(),
}
```

### 添加新的损失函数

1. 在 `custom_components/losses/custom_losses.py` 中创建新类：

```python
class YourCustomLoss(BaseCustomLoss):
    def create_loss(self, **kwargs):
        # 实现你的损失函数逻辑
        return YourLossFunction(**kwargs)
    
    def get_default_config(self):
        return {
            "reduction": "mean",
            # 其他默认参数
        }
```

2. 在 `CUSTOM_LOSSES` 字典中注册：

```python
CUSTOM_LOSSES = {
    # 现有损失函数...
    "your_loss": YourCustomLoss(),
}
```

## 注意事项

1. **兼容性**: 自定义组件与现有的训练流程完全兼容
2. **回退机制**: 如果指定的自定义组件不存在，系统会自动回退到内置组件
3. **参数传递**: 所有在 `train_parameters` 中的参数都会传递给相应的组件
4. **错误处理**: 组件初始化失败时会有详细的错误信息

## 最佳实践

1. **命名规范**: 使用描述性的名称，避免与内置组件冲突
2. **参数验证**: 在组件中添加适当的参数验证
3. **文档说明**: 为新组件添加清晰的文档字符串
4. **测试**: 创建简单的测试来验证组件功能

## 故障排除

### 常见问题

1. **导入错误**: 确保项目路径正确添加到 `sys.path`
2. **参数错误**: 检查传递给组件的参数是否正确
3. **版本兼容**: 确保PyTorch版本与组件实现兼容

### 调试技巧

1. 使用示例代码测试组件
2. 检查错误日志中的详细信息
3. 验证组件注册是否正确