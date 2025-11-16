# 密码算法输出预测项目 (Output Prediction)

本项目使用深度学习技术研究“输出预测攻击”在分组密码与流密码上的可行性与效果，支持基于统一配置的超参数搜索与训练评估。

## 项目概述

- 生成密码算法的明文/密文数据集（支持分批次与断点续传）
- 使用统一的训练器进行模型训练、验证与测试
- 通过 Optuna 进行超参数搜索优化
- 输出标准评估指标与可选训练过程可视化

## 项目结构

```
Output_Prediction/
├── ciphers/                         # 密码算法实现（分组与流密码）
│   ├── base_cipher.py               # 抽象基类
│   ├── present.py                   # SmallPRESENT4 及相关变体
│   ├── aes.py, AES64.py, AES128.py  # 简化 AES 族
│   ├── twine.py                     # TWINE 简化实现
│   ├── grain128a.py                 # Grain-128a 流密码
│   ├── trivium.py                   # Trivium 流密码
│   ├── acornv3.py                   # Acorn v3 流密码
│   └── zuc256.py                    # ZUC-256 流密码
├── experiments/                     # 实验脚本
│   ├── exp1_simplify.py             # 超参数优化（统一入口）
│   ├── exp2_simplify.py             # 攻击成功率验证（统一入口）
│   └── exp3_variants.py             # 密码变体攻击
├── models/                          # 模型与训练
│   ├── lstm_model.py                # LSTM 模型
│   ├── resnet_model.py              # ResNet 风格模型
│   ├── mamba_model.py               # Mamba 序列模型
│   ├── iTransformer.py              # iTransformer 模型
│   └── train_model_simplify.py      # 统一训练器
├── callbacks/                       # 训练回调
│   ├── plotting.py                  # 训练过程绘图
│   ├── metrics_tracker.py           # 指标采集
│   └── early_stopping.py            # 早停（按需）
├── utils/                           # 工具与辅助
│   ├── data_generator.py            # 数据生成
│   ├── metrics.py                   # 评估指标
│   ├── directory_manager.py         # 目录管理
│   └── config.py                    # 统一配置读取（JSON 驱动）
├── configs/config.json              # 外部化统一配置
├── requirements.txt                 # 依赖说明（按需安装）
└── main.py                          # 命令行入口
```

## 安装与环境

- Python 3.10+（推荐）
- 按需安装 PyTorch（CPU/GPU 版本）
- 核心依赖见 `requirements.txt`：`numpy`、`einops`、`optuna`、`matplotlib`

示例安装：
```bash
pip install -r requirements.txt
# 按需安装 PyTorch：
# CPU 版
pip install torch==2.1.*
# 或 GPU 版（以 CUDA 11.8 为例）
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.*
```

检查 GPU：
```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

## 统一配置

项目通过 `configs/config.json` 外部化全部关键配置，包括：
- `MODEL_TYPE`：选择模型类型（`lstm`、`resnet`、`mamba`、`itransformer`）
- `EPOCHS`、`BLOCK_SIZE`、`KEY_SIZE` 等训练与数据维度参数
- `HYPERPARAMETER_SEARCH_SPACE`：Optuna 搜索空间
- `CIPHER_REGISTRY`：算法注册表（名称 → 模块路径与类名）
- `BEST_HYPERPARAMETERS`：各算法的已验证最优超参数
- `EXPERIMENT_CONFIGS`：各实验的数据规模与标签选择

支持通过环境变量覆盖配置路径：
```bash
export OUTPUT_PREDICTION_CONFIG_PATH="/abs/path/to/config.json"
```

## 快速使用

使用统一入口 `main.py` 运行实验：
```bash
# 实验1：超参数优化并最终训练
python main.py --experiment exp1 --cipher present --rounds 4 --mode all

# 实验2：攻击成功率验证（端到端）
python main.py --experiment exp2 --cipher present --rounds 4 --mode all

# 实验3：密码变体攻击（数据+训练+运行）
python main.py --experiment exp3 --cipher present --rounds 4 --mode all
```

按需仅生成数据或仅训练：
```bash
# 仅数据采集
python main.py --experiment exp1 --cipher aes --rounds 4 --mode data

# 仅训练
python main.py --experiment exp2 --cipher trivium --rounds 4 --mode train
```

可选子命令（示例）：
```bash
# exp2_simplify 独立运行
python experiments/exp2_simplify.py generate --cipher present --rounds 4
python experiments/exp2_simplify.py train --cipher present --rounds 4 --disable-plotting
```

## 数据生成与输出

- 数据生成使用 `utils/data_generator.py` 的批次化与断点续传机制，目录示例：
  - `data/exp1_train_present_round4/`
    - `batches/`（中间批次文件，自动清理）
    - `plain_texts.npy`、`cipher_texts.npy`
- 训练与测试结果默认输出到 `results/exp1`、`results/exp2`、`results/exp3`（如启用保存）
- 训练过程图表将保存到 `plots/`（由回调自动管理）

## 评估指标

- `bitwise_success_rate`：逐比特成功率
- `sample_success_rate`：样本级完全匹配成功率
- `log2_success_rate`：二进制对数刻度成功率
- `bit_match_percentage`：比特匹配百分比

## 密码算法支持

根据注册表与实现，当前支持：
- 分组密码：`present`、`aes`、`aes64`、`aes128`、`twine`
- 流密码：`grain128a`、`trivium`、`acornv3`、`zuc256`
- 变体：`weak_S_box1_present`、`weak_S_box2_present`、`component_modification_present`

## 模块示例

使用 SmallPRESENT4 加密（最小示例）：
```python
from ciphers.present import SmallPRESENT4
import numpy as np

sp = SmallPRESENT4(rounds=4)
pt = np.random.randint(0, 2, sp.block_size, dtype=np.uint8)
key = np.random.randint(0, 2, sp.key_size, dtype=np.uint8)
ct = sp.encrypt(pt, key)
print("ciphertext bits:", ct)
```

数据集生成（标签可选范围字符串）：
```python
from utils.data_generator import generate_dataset
from utils.config import config

cipher = config.getCipherInstance("present", 4, config.getBlockSize(), config.getKeySize())
generate_dataset(
    cipher=cipher,
    num_keys=10,
    total_data=5000,
    save_dir="./data/exp1_train_present_round4",
    target_index="112-127",  # 选择比特范围作为标签
    batch_size=500,
    resume=True,
)
```

## 常见问题

- 训练时间长：减少 `EPOCHS`、降低模型复杂度或使用 GPU。
- 内存不足：减小 `batch_size`、降低样本数量，或启用分批次生成。
- 准确率偏低：增大训练数据、调整 `MODEL_TYPE` 与超参数，运行 `exp1` 搜索。
- 配置未生效：检查 `OUTPUT_PREDICTION_CONFIG_PATH` 或 `configs/config.json` 路径。

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE`。

## 更新日志

### v2.1 (2025-10-25)
- 引入统一 JSON 配置系统（`configs/config.json`）
- 新增统一训练器 `models/train_model_simplify.py`
- 替换与简化 `exp1/exp2` 运行入口（`exp1_simplify.py`、`exp2_simplify.py`）
- 扩展密码算法（AES64、AES128、Grain-128a、Trivium、Acorn v3、ZUC-256）
- 增强数据生成：支持批次化与断点续传；标签范围字符串

---

本项目仅用于教育与研究目的，不应用于实际密码系统安全评估。
