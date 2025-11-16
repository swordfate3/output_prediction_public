# 简化版训练模型 (TrainModelSimplify)

这是一个简化版的训练模型，保留了核心训练功能并支持绘图回调，适合快速原型开发和实验。

## 特点

- **简单易用**: 去除了复杂的早停、自定义组件等高级功能
- **核心功能完整**: 保留了训练、验证、测试的核心功能
- **绘图支持**: 集成了绘图回调，自动生成训练曲线图
- **快速原型开发**: 适合快速实验和原型开发
- **清晰的代码结构**: 代码逻辑简单明了，易于理解和修改

## 主要功能

### 1. 基本训练循环
- 标准的训练/验证循环
- 自动计算损失和准确率
- 支持GPU加速

### 2. 数据处理
- 自动加载训练和测试数据
- 自动划分验证集
- 张量转换和数据加载器创建

### 3. 模型评估
- 计算准确率、比特成功率、Log2成功率
- 支持测试集评估

### 4. 绘图功能 ✨
- 自动生成训练曲线图
- 包含损失和准确率曲线
- 图片保存到 plots/ 目录
- 可选择启用/禁用

### 5. 模型保存/加载
- 简单的模型保存和加载功能

## 使用方法

### 基本使用

```python
from models.train_model_simplify import TrainModelSimplify

# 1. 初始化训练器（启用绘图功能）
trainer = TrainModelSimplify(
    hidden_dim=128,      # LSTM隐藏层维度
    num_layers=2,        # LSTM层数
    lr=0.001,           # 学习率
    optimizer="Adam",   # 优化器
    criterion="BCELoss",# 损失函数
    threshold=0.5,       # 概率阈值（用于二值化预测，支持超参数搜索）
    batch_size=32,      # 批次大小
    epochs=50,          # 训练轮数
    enable_plotting=True, # 启用绘图功能
    model_name="LSTM"    # 模型选择："LSTM"、"Mamba" 或 "ResNet"
)

# 2. 加载数据
train_loader, val_loader, test_loader = trainer.load_data(
    data_dir="./data",
    cipher_name="present",
    rounds=4,
    data_prefix="exp1"
)

# 3. 训练模型（自动生成训练曲线图）
train_results = trainer.train(train_loader, val_loader)

# 4. 测试模型
test_results = trainer.test(test_loader)

# 5. 保存模型
trainer.save_model("my_model.pth")
```

### 禁用绘图功能

如果不需要绘图功能，可以在初始化时禁用：

```python
trainer = TrainModelSimplify(
    hidden_dim=128,
    num_layers=2,
    lr=0.001,
    batch_size=32,
    epochs=50,
    enable_plotting=False,  # 禁用绘图功能
    model_name="Mamba"      # 例：选择使用 Mamba 模型
)
```

### 参数说明

#### 初始化参数
- `hidden_dim`: LSTM隐藏层维度 (默认: 64)
- `num_layers`: LSTM层数 (默认: 2)
- `lr`: 学习率 (默认: 0.001)
- `optimizer`: 优化器名称 (默认: `Adam`，支持: `SGD`, `Adam`, `RMSprop`)
- `criterion`: 损失函数名称 (默认: `BCELoss`，支持: `BCELoss`, `MSELoss`)
- `batch_size`: 批次大小 (默认: 32)
- `epochs`: 训练轮数 (默认: 100)
- `enable_plotting`: 是否启用绘图功能 (默认: True) ✨
- `model_name`: 模型名称 (默认: `"LSTM"`，可选 `"Mamba"`, `"ResNet"`)。
  - 选择 `"Mamba"` 时使用 `CipherMamba` 时序模型（其内部 `d_state/d_conv/expand/dropout` 使用默认值）。
  - 选择 `"ResNet"` 时使用 `CipherResNet` 前馈残差网络（线性-BN-ReLU 残差块堆叠，适合固定维度输入到概率输出）。

#### 数据加载参数
- `data_dir`: 数据目录路径
- `cipher_name`: 密码算法名称 (如 "present")
- `rounds`: 密码轮数 (默认: 4)
- `data_prefix`: 数据前缀 (默认: "exp1")
- `val_split`: 训练集比例 (默认: 0.8)

## 绘图功能说明 ✨

### 自动生成的图表
- **训练曲线图**: 包含训练和验证的损失、准确率曲线
- **文件格式**: PNG格式，300 DPI高清图片
- **保存位置**: `plots/` 目录下
- **文件命名**: 基于实验名称、密码算法和轮数自动命名

### 图表内容
- 训练损失 vs 验证损失
- 训练准确率 vs 验证准确率
- 清晰的图例和标签
- 专业的图表样式

### 绘图时机
- 训练结束时自动生成最终图表
- 如果matplotlib不可用，会自动禁用绘图功能并给出提示

## 返回结果

### 训练结果 (train_results)
```python
{
    "train_losses": [...],           # 训练损失历史
    "val_losses": [...],             # 验证损失历史
    "train_accs": [...],             # 训练准确率历史
    "val_accs": [...],               # 验证准确率历史
    "final_val_acc": 0.xxxx,         # 最终验证准确率
    "final_bitwise_sr": 0.xxxx,      # 最终比特成功率
    "final_log2_sr": 0.xxxx          # 最终Log2成功率
}
```

### 测试结果 (test_results)
```python
{
    "test_loss": 0.xxxx,             # 测试损失
    "test_acc": 0.xxxx,              # 测试准确率
    "test_bitwise_sr": 0.xxxx,       # 测试比特成功率
    "test_log2_sr": 0.xxxx           # 测试Log2成功率
}
```

## 与完整版本的区别

| 功能 | 简化版 | 完整版 |
|------|--------|--------|
| 基本训练循环 | ✅ | ✅ |
| 数据加载 | ✅ | ✅ |
| 模型评估 | ✅ | ✅ |
| 绘图回调 | ✅ | ✅ |
| 早停回调 | ❌ | ✅ |
| 自定义组件 | ❌ | ✅ |
| 复杂错误处理 | ❌ | ✅ |
| 详细日志记录 | ❌ | ✅ |
| 梯度范数计算 | ❌ | ✅ |

## 运行示例

直接运行文件可以看到使用示例：

```bash
cd /path/to/Output_Prediction
python models/train_model_simplify.py
```

运行后会在 `plots/` 目录下生成训练曲线图。

## 注意事项

1. **数据路径**: 确保数据文件存在于正确的路径中
2. **matplotlib依赖**: 绘图功能需要matplotlib，如果未安装会自动禁用绘图
3. **内存使用**: 简化版本没有复杂的内存管理，大数据集时需要注意内存使用
4. **错误处理**: 简化版本的错误处理较为基础，遇到问题时可能需要手动调试
5. **扩展性**: 如果需要早停等高级功能，建议使用完整版本的 `TrainModel`

## 适用场景

- 快速原型开发
- 简单实验验证
- 学习和理解训练流程
- 需要可视化训练过程的基础训练任务
- 不需要复杂功能但需要结果可视化的场景