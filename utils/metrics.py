import numpy as np
import torch


def bitwise_success_rate(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """计算每bit的平均成功率（辅助指标），提高数值稳定性。

    - 将预测与目标转换为整型后比较，避免浮点比较误差
    - 使用 64 位整型累计正确位数，再以双精度浮点返回比例
    """
    preds_i = preds.to(dtype=torch.int8)
    targets_i = targets.to(dtype=torch.int8)
    correct = (preds_i == targets_i).to(torch.int64).sum().item()
    total = targets_i.numel()
    return float(correct) / float(total)


def log2_success_rate(success_rate: float) -> float:
    """将成功率转换为 2 的负对数形式（论文表达方式）。

    - 对 0 返回正无穷
    - 使用双精度进行计算
    """
    if success_rate == 0.0:
        return np.inf
    return float(-np.log2(float(success_rate)))


def sample_success_rate(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算样本级完全匹配成功率（Exact Match Success Rate），提高数值精度。

    - 对每个样本，所有位均与目标一致才计为成功
    - 使用整型比较避免浮点误差；以 64 位整型计数成功样本数
    - 返回双精度浮点比例（成功样本数 / 总样本数）
    """
    preds_i = preds.to(dtype=torch.int8)
    targets_i = targets.to(dtype=torch.int8)

    if preds_i.dim() >= 2:
        exact_match = (preds_i == targets_i).all(dim=-1)
    else:
        exact_match = (preds_i == targets_i)

    successes = torch.count_nonzero(exact_match).item()
    total = exact_match.numel()
    return float(successes) / float(total)


def bit_match_percentage(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算位数匹配百分比（每个样本的匹配位占比的平均值）。

    - 对每个样本，统计与目标相同的位数，并除以总位数得到百分比
    - 返回所有样本百分比的平均值（介于 0.0 与 1.0 之间）
    - 示例：16 位中有 8 位匹配，则该样本为 0.5；整批返回它们的平均值

    Args:
        preds: 预测张量，形状通常为 (B, D) 或 (B, 1, D)
        targets: 目标张量，形状与 preds 对齐

    Returns:
        float: 平均位匹配百分比（0.0~1.0）
    """
    # 统一为整型比较，避免浮点误差
    preds_i = preds.to(dtype=torch.int8)
    targets_i = targets.to(dtype=torch.int8)

    # 若存在额外的序列维度，去除它（与模型输出规范保持一致）
    if preds_i.dim() == 3 and preds_i.shape[1] == 1:
        preds_i = preds_i.squeeze(1)
    if targets_i.dim() == 3 and targets_i.shape[1] == 1:
        targets_i = targets_i.squeeze(1)

    # 按最后一维（位维度）计算每样本匹配位数
    matches_per_sample = (preds_i == targets_i).to(torch.int64).sum(dim=-1)
    total_bits = preds_i.shape[-1]
    # 避免除 0（理论上不应出现 D=0）
    if total_bits <= 0:
        return 0.0

    percentages = matches_per_sample.to(dtype=torch.float64) / float(total_bits)
    # 返回样本百分比的平均值
    return float(percentages.mean().item())