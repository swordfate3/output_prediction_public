# iTransformer 烟雾测试报告

本报告记录了对 `/models/iTransformer.py` 中 `CipherITransformer` 的快速烟雾测试结果，用于验证模型在构造、前向、反向传播以及保存/加载关键路径上的正确性与稳定性。测试脚本为一次性使用，已在测试结束后删除，仅保留本报告用于讲解与复现参考。

> [ADD] 已新增一次性测试脚本进行快速验证；[DEL] 测试完成后删除临时测试脚本与检查点文件。

## 测试环境与方法
- 环境：`Linux`，`Python`，`PyTorch`
- 被测类：`models/iTransformer.py` 中的 `CipherITransformer`
- 测试范围：
  - 构造与参数传入
  - 前向传播（两种输入形状）
  - 反向传播（一次优化步）
  - 保存与加载检查点的一致性

## 采用参数
- `input_dim`: `16`
- `hidden_dim`: `64`
- `num_layers`: `2`
- `num_heads`: `4`
- `output_dim`: `16`
- `dropout`: `0.1`

## 测试结果
- 前向传播：
  - 输入形状 `(8, 1, 16)` 输出形状 `[8, 16]`，范围有效 `[0, 1]`
    - `min`: `0.24247`，`max`: `0.79013`，`mean`: `0.49709`
  - 输入形状 `(8, 16)` 输出形状 `[8, 16]`，范围有效 `[0, 1]`
    - `min`: `0.24098`，`max`: `0.79013`，`mean`: `0.49717`
- 反向传播：
  - `loss_before`: `0.80581`
  - `loss_after`: `0.70620`
  - 单步训练耗时：约 `0.068 s`
- 保存/加载：
  - `loaded_ok`: `true`（加载后输出与保存前一致）
  - `has_config`: `true`
  - `has_state`: `true`

结论：在上述参数配置与输入条件下，模型关键路径均正常，未发现阻断性错误。

## 潜在问题与建议
- 位置编码初始化：`pos_embed` 为可学习参数，初始为零。这是可行的，但可能导致训练前期表达能力有限。建议尝试：
  - 使用正弦位置编码或将 `pos_embed` 以截断正态分布初始化。
- 损失函数选择：当前模型头部含 `Sigmoid`，烟雾测试使用 `BCELoss`。若训练过程中使用 `BCEWithLogitsLoss`，则应移除 `Sigmoid`（或切换为 `BCELoss` 保持一致）。请确保训练管线一致性。
- 保存到当前目录的健壮性：`saveModel` 会调用 `os.makedirs(os.path.dirname(filepath), exist_ok=True)`。当 `filepath` 不含目录（如仅 `checkpoint.pth`）时，`dirname` 为空字符串，可能在部分环境下引发异常。建议在创建目录前判断非空。
- 可扩展性：`dim_feedforward=hidden_dim * 4` 为固定设置，若进行超参搜索可考虑暴露该倍率参数；同时可加入注意力掩码支持以适配变长输入。

## 复现与使用提示
- 若需快速自检，可参考以下最小示例：
  ```python
  import torch
  from models.iTransformer import CipherITransformer

  model = CipherITransformer(input_dim=16, hidden_dim=64, num_layers=2,
                             num_heads=4, output_dim=16, dropout=0.1)
  x1 = torch.randint(0, 2, (8, 1, 16)).float()
  x2 = torch.randint(0, 2, (8, 16)).float()
  y1 = model.eval()(x1)
  y2 = model.eval()(x2)
  print(y1.shape, y2.shape)
  ```

## 清理说明
- [DEL] 一次性测试脚本 `tests/test_itransformer_smoke.py` 已删除。
- [DEL] 测试过程中生成的临时检查点文件已在脚本执行期间清理。

## 结语
整体来看，`CipherITransformer` 在标准配置下行为稳定、接口清晰。若后续需要深入评估泛化性能与训练稳定性，建议结合项目的超参数搜索与真实数据集开展系统性实验，并统一训练管线中的损失函数与输出头设计。