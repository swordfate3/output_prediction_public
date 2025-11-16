#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练模型

提供基本的训练、验证和测试功能，支持绘图回调
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import CipherLSTM
from models.mamba_model import CipherMamba
from models.resnet_model import CipherResNet
# [ADD] 导入 CipherITransformer 以在训练脚本中选择使用
from models.iTransformer import CipherITransformer
from utils.metrics import (
    bitwise_success_rate,
    log2_success_rate,
    sample_success_rate,
    bit_match_percentage,
)
from callbacks.plotting import PlottingCallback
from utils.directory_manager import get_plots_directory
from utils.logger import getGlobalLogger
import time

logger = getGlobalLogger()


class TrainModelSimplify:
    """
    简化版训练模型类

    提供基本的模型训练、验证和测试功能，支持绘图回调
    """

    def __init__(
        self,
        hidden_dim=64,
        num_layers=2,
        lr=0.001,
        batch_size=32,
        epochs=100,
        optimizer="Adam",
        criterion="BCELoss",
        threshold: float = 0.5,
        enable_plotting=True,
        model_name: str = "LSTM",
        dropout: float | None = None,
        model_params: dict | None = None,
        # [ADD] 控制测试成功率柱状图是否使用对数刻度放大显示
        plot_sr_log_scale: bool = False,
        # [ADD] 仅对 Sample SR 使用右侧对数刻度的混合显示
        plot_sr_mixed_scale: bool = False,
    ):
        """
        初始化简化训练模型

        详细描述：该构造函数接收通用训练与模型控制参数，并支持以顶层参数
        传入通用的 `dropout`，由训练器在不同模型构造时统一使用；模型特定
        参数（如 `num_heads`、`d_state`）通过 `model_params` 传入。

        Args:
            hidden_dim (int): 隐藏层维度（适用于多数模型）
            num_layers (int): 网络层数/重复块数
            lr (float): 学习率
            batch_size (int): 批次大小
            epochs (int): 训练轮数
            optimizer (str): 优化器名称（支持 "SGD"、"Adam"、"RMSprop"）
            criterion (str): 损失函数名称（支持 "BCELoss"、"MSELoss"）
            threshold (float): 概率二值化阈值（计算准确率/成功率指标时使用）
            enable_plotting (bool): 是否启用绘图功能
            model_name (str): 模型名称（支持 "LSTM"、"Mamba"、"ResNet"、"iTransformer"）
            dropout (float | None): 通用丢弃率；若提供，将用于所有支持该参数的模型
            model_params (dict | None): 模型特定的可选参数字典；如：
                - Mamba: {"d_state": 16, "d_conv": 4, "expand": 2}
                - iTransformer: {"num_heads": 4}

        Returns:
            None: 构造函数不返回值，完成内部状态初始化

        Raises:
            ValueError: 当 `criterion` 或 `optimizer` 非支持集合时抛出

        Example:
            >>> trainer = TrainModelSimplify(
            ...     hidden_dim=192, num_layers=4, lr=0.001,
            ...     batch_size=250, epochs=100, optimizer="Adam",
            ...     criterion="BCELoss", model_name="iTransformer",
            ...     dropout=0.1, model_params={"num_heads": 4}
            ... )
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.criterion_name = criterion
        self.threshold = threshold
        self.enable_plotting = enable_plotting
        self.model_name = model_name
        # [ADD] 顶层通用 dropout；若为 None，按模型默认或 model_params 回退
        self.dropout = dropout
        # [ADD] 记录模型特定参数（若未提供则置为空字典）
        self.model_params = dict(model_params or {})
        # [ADD] 成功率柱状图对数刻度开关，解决 Sample SR 很小难以观察的问题
        self.plot_sr_log_scale = plot_sr_log_scale
        # [ADD] 仅对 Sample SR 使用对数刻度的混合显示开关
        self.plot_sr_mixed_scale = plot_sr_mixed_scale

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型组件
        self.model = None
        self.optimizer = None
        # 选择损失函数
        crit_map = {
            "BCELoss": nn.BCELoss,
            "MSELoss": nn.MSELoss,
        }
        if self.criterion_name not in crit_map:
            raise ValueError(
                f"不支持的损失函数: {self.criterion_name}，支持: "
                f"{list(crit_map.keys())}"
            )
        self.criterion = crit_map[self.criterion_name]()

        # 训练历史和结果（兼容绘图回调）
        self.training_results = {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "bitwise_success_rates": [],
            "log2_success_rates": [],
            "sample_success_rates": [],
            "bit_match_percentages": [],
            "total_epochs": 0,
        }

        # 绘图回调
        self.plotting_callback = None

    def load_data(
        self,
        data_dir,
        cipher_name,
        rounds=4,
        data_prefix="exp1",
        val_split=0.8,
        test_split=0.1,
        derive_test_from_train=True,
    ):
        """
        加载训练、验证与测试数据

        Args:
            data_dir (str): 数据目录
            cipher_name (str): 密码算法名称
            rounds (int): 密码轮数
            data_prefix (str): 数据前缀
            val_split (float): 训练集比例（历史命名保留；当 derive_test_from_train=True 时忽略）
            test_split (float): 测试集比例（例如0.1表示10%用于测试）
            derive_test_from_train (bool): 是否从训练数据切分得到测试集；
                为 True 时先按 test_split 划出测试集，剩余作为“训练池”，
                验证集固定为训练池的 20%，训练集为训练池的 80%

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # 构建数据路径
        train_path = f"{data_dir}/{data_prefix}_train_{cipher_name}_round{rounds}"
        test_path = f"{data_dir}/{data_prefix}_test_{cipher_name}_round{rounds}"

        # 加载训练数据（作为整体数据源）
        X_all = np.load(f"{train_path}/plain_texts.npy")
        y_all = np.load(f"{train_path}/cipher_texts.npy")

        if derive_test_from_train:
            # 先划出测试集；再让验证集占训练池的20%，训练占80%
            n = len(X_all)
            test_ratio = float(test_split)
            if test_ratio < 0 or test_ratio > 1:
                raise ValueError("test_split 必须在 [0, 1] 范围内")

            test_len = int(n * test_ratio)
            # 训练池 = 全部样本 - 测试集
            train_pool_len = max(0, n - test_len)
            # 验证集固定为训练池的 20%
            val_len = int(train_pool_len * 0.2)
            # 实际训练集为训练池剩余的 80%
            train_len = max(0, train_pool_len - val_len)

            # 切片分配：Train | Val | Test（保证不重叠，总长不超过 n）
            X_train = X_all[:train_len]
            y_train = y_all[:train_len]
            X_val = X_all[train_len:train_len + val_len]
            y_val = y_all[train_len:train_len + val_len]
            X_test = X_all[train_len + val_len:train_len + val_len + test_len]
            y_test = y_all[train_len + val_len:train_len + val_len + test_len]
        else:
            # 保持原有行为：从独立测试路径加载，训练集仅切分验证
            X_train = X_all
            y_train = y_all
            X_test = np.load(f"{test_path}/plain_texts.npy")
            y_test = np.load(f"{test_path}/cipher_texts.npy")
            # 划分训练和验证集
            split_idx = int(len(X_train) * val_split)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

        # 转换为张量
        X_train = torch.FloatTensor(X_train).unsqueeze(1)  # 添加序列维度
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val).unsqueeze(1)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_test = torch.FloatTensor(y_test)

        # 若目标为一维（每样本一个标量），扩展为 (N, 1)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(-1)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(-1)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # 初始化模型
        input_dim = X_train.shape[-1]
        # 输出维度应为目标的最后一维；若原始目标为标量，已上移为 (N, 1)
        output_dim = y_train.shape[-1]
        self._init_model(input_dim, output_dim)

        # 初始化绘图回调
        if self.enable_plotting:
            self._init_plotting_callback(cipher_name, rounds, data_prefix)

        return train_loader, val_loader, test_loader

    def _init_model(self, input_dim, output_dim):
        """初始化模型与优化器

        使用传入的输入/输出维度以及类初始化参数（隐藏维度、层数、优化器名称等）
        构建指定类型的模型实例，并据此创建优化器。

        Args:
            input_dim (int): 模型输入特征维度（通常为比特位数，如16或128）
            output_dim (int): 模型输出维度（通常为目标比特位数）

        Returns:
            None: 方法内部完成模型与优化器的构建与赋值

        Example:
            >>> self.model_name = "iTransformer"
            >>> self._init_model(input_dim=16, output_dim=16)
            >>> isinstance(self.model, CipherITransformer)
            True
        """
        name = (self.model_name or "LSTM").lower()
        # [ADD] 统一读取模型可选参数，避免多行续行缩进问题（flake8 E128）
        params = self.model_params or {}
        # [MOD] 预计算不同模型使用的可选参数；优先使用顶层通用 dropout
        num_heads_transformer = params.get("num_heads", 4)
        dropout_resnet = self.dropout if self.dropout is not None else params.get("dropout", 0.1)
        dropout_transformer = self.dropout if self.dropout is not None else params.get("dropout", 0.1)
        dropout_lstm = self.dropout if self.dropout is not None else params.get("dropout", 0.2)
        mamba_d_state = params.get("d_state", 16)
        mamba_d_conv = params.get("d_conv", 4)
        mamba_expand = params.get("expand", 2)
        mamba_dropout = self.dropout if self.dropout is not None else params.get("dropout", 0.1)
        if name == "mamba":
            # [MOD] 统一在此分支使用可选模型参数，支持 d_state/d_conv/expand/dropout
            self.model = CipherMamba(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=output_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
            ).to(self.device)
        elif name == "resnet":
            # 使用残差前馈网络，适合非序列特征到概率输出
            self.model = CipherResNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_layers=self.num_layers,
                # [MOD] 简化参数传递：统一使用预计算的变量，修复续行缩进问题
                dropout=dropout_resnet,
            ).to(self.device)
        elif name in ("itransformer", "transformer"):
            # [ADD] 使用简化版 Transformer 模型 CipherITransformer（与其他模型统一的 forward(x) 接口）
            # 说明：输入按比特序列处理；输出为每位的概率，最终训练用 BCELoss/MSELoss 均可
            self.model = CipherITransformer(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=output_dim,
                # [MOD] 简化参数传递：统一使用预计算的变量，修复续行缩进问题
                num_heads=num_heads_transformer,
                dropout=dropout_transformer,
            ).to(self.device)
        elif name == "lstm":
            # 默认使用 LSTM 模型
            self.model = CipherLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_layers=self.num_layers,
                # [MOD] 简化参数传递：统一使用预计算的变量，修复续行缩进问题
                dropout=dropout_lstm,
            ).to(self.device)
        # [DEL] 删除：重复的 mamba 分支，已统一于首个 mamba 分支处理可选参数

        # 选择优化器
        opt_map = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }
        if self.optimizer_name not in opt_map:
            raise ValueError(
                f"不支持的优化器: {self.optimizer_name}，支持: {list(opt_map.keys())}"
            )
        OptimizerCls = opt_map[self.optimizer_name]
        # 对于不同优化器可补充默认参数，这里统一仅使用学习率
        self.optimizer = OptimizerCls(self.model.parameters(), lr=self.lr)

    def _normalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """规范化模型输出形状以匹配目标维度。

        - 对于序列输出 (B, 1, D)，去除 seq 维度，得到 (B, D)
        - 其它形状保持不变
        """
        if outputs.dim() == 3 and outputs.shape[1] == 1:
            return outputs.squeeze(1)
        return outputs

    def _init_plotting_callback(self, cipher_name, rounds, experiment_name):
        """初始化绘图回调"""
        try:
            self.plotting_callback = PlottingCallback(
                experiment_name=experiment_name, cipher_name=cipher_name, rounds=rounds
            )
            # 设置训练器引用，让回调能访问训练结果
            self.plotting_callback.set_trainer(self)
            print("✓ 绘图功能已启用")
        except Exception as e:
            print(f"⚠ 绘图功能初始化失败: {e}")
            self.plotting_callback = None

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = self._normalize_outputs(outputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > self.threshold).float()
            correct += (predicted == targets).sum().item()
            total += targets.numel()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = (inputs.to(self.device), targets.to(self.device))

                outputs = self.model(inputs)
                outputs = self._normalize_outputs(outputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                predicted = (outputs > self.threshold).float()
                correct += (predicted == targets).sum().item()
                total += targets.numel()

                all_predictions.append(predicted.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        # 计算成功率指标
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        bitwise_sr = bitwise_success_rate(predictions, targets)
        log2_sr = log2_success_rate(bitwise_sr)
        sample_sr = sample_success_rate(predictions, targets)
        bit_match_pct = bit_match_percentage(predictions, targets)

        return avg_loss, accuracy, bitwise_sr, log2_sr, sample_sr, bit_match_pct

    def _plot_test_curves(self, test_loss, test_acc,
                          bitwise_sr, log2_sr,
                          sample_sr, bit_match_pct):
        """
        绘制并保存测试集整体指标图（不按批次）。

        详细描述：训练完成的测试评估将返回整体平均指标，本函数将其可视化：
        - 左侧文本摘要：Loss、Accuracy、Log2 SR
        - 右侧条形图：Bitwise SR、Sample SR、Bit Match %（统一到 [0,1]）

        Args:
            test_loss (float): 测试集平均损失
            test_acc (float): 测试集平均准确率（位级）
            bitwise_sr (float): 位级成功率（整体）
            log2_sr (float): 成功率的 -log2 表达（整体）
            sample_sr (float): 样本级成功率（整体）
            bit_match_pct (float): 位匹配百分比（整体）

        Returns:
            None: 保存图片到 plots 目录

        Raises:
            Exception: 保存或绘制失败时捕获并打印提示

        Example:
            >>> self._plot_test_curves(0.25, 0.82, 0.80, 0.32, 0.12, 0.80)
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # [MOD] 切换为整体摘要 + 成功率条形图
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 左：文本摘要
            axes[0].axis('off')
            summary_lines = [
                f"Test Loss: {float(test_loss):.4f}",
                f"Test Accuracy: {float(test_acc):.4f}",
                f"Log2 Success Rate (-log2 p): {float(log2_sr):.4f}",
            ]
            text_y = 0.8
            for line in summary_lines:
                axes[0].text(0.05, text_y, line, fontsize=12,
                             transform=axes[0].transAxes)
                text_y -= 0.12
            axes[0].set_title("Test Summary", fontsize=14)

            # 右：成功率条形图
            sr_labels = ["Bitwise SR", "Sample SR", "Bit Match %"]
            sr_values = [float(bitwise_sr), float(sample_sr), float(bit_match_pct)]
            # [ADD] 混合刻度：仅对 Sample SR 使用右侧对数刻度
            if self.plot_sr_mixed_scale:
                import numpy as np
                positions = np.arange(len(sr_labels))
                axes[1].set_xticks(positions)
                axes[1].set_xticklabels(sr_labels)
                axes[1].bar([positions[0], positions[2]],
                            [sr_values[0], sr_values[2]],
                            color=["purple", "blue"]) 
                axes[1].set_ylim(0.0, 1.0)
                axes[1].set_ylabel("Rate")
                axes[1].set_title("Test Success Rates (Mixed Axes)")
                axes[1].grid(True, axis='y', alpha=0.3)

                ax_r = axes[1].twinx()
                sample_val = sr_values[1]
                # [MOD] 使用右轴线性刻度，便于无需log也可放大显示小值
                # 默认将右轴上限设为1.0，下限自适应在样本值附近
                upper = 1.0
                lower = max(0.0, sample_val * 0.8)
                ax_r.bar([positions[1]], [sample_val], color=["orange"]) 
                ax_r.set_ylim(lower, upper)
                ax_r.set_ylabel("Sample SR (linear)")
                # 数值标注
                axes[1].text(positions[0], sr_values[0] + 0.02,
                             f"{sr_values[0]:.3f}", ha='center', va='bottom', fontsize=10)
                axes[1].text(positions[2], sr_values[2] + 0.02,
                             f"{sr_values[2]:.3f}", ha='center', va='bottom', fontsize=10)
                ax_r.text(positions[1], sample_val * 1.05,
                          f"{sample_val:.3f}", ha='center', va='bottom', fontsize=10)
            # [ADD] 对数刻度支持：当 Sample SR 很小（例如 1e-6）时更易观察
            elif self.plot_sr_log_scale:
                # [ADD] 设定对数刻度的下界，避免 0 值导致 -inf
                eps = 1e-8
                min_pos = min([v for v in sr_values if v > 0] or [eps])
                axes[1].set_yscale('log')
                axes[1].bar(sr_labels, [max(v, eps) for v in sr_values],
                            color=["purple", "orange", "blue"])
                axes[1].set_ylim(max(eps, min_pos * 0.8), 1.0)
                axes[1].set_ylabel("Rate (log)")
                axes[1].set_title("Test Success Rates (Log Scale)")
            else:
                axes[1].bar(sr_labels, sr_values, color=["purple", "orange", "blue"])
                axes[1].set_ylim(0.0, 1.0)
                axes[1].set_ylabel("Rate")
                axes[1].set_title("Test Success Rates (Overall)")
            axes[1].grid(True, axis='y', alpha=0.3)

            save_dir = get_plots_directory(create=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            if self.plotting_callback:
                exp = self.plotting_callback.experiment_name
                cipher = self.plotting_callback.cipher_name
                rounds = self.plotting_callback.rounds
                filename = f"{exp}_test_curves_{cipher}_round{rounds}_{ts}.png"
            else:
                filename = f"test_curves_{ts}.png"

            import os

            path = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            print(f"✓ 测试曲线图已保存: {path}")
        except Exception as e:
            print(f"⚠ 保存测试曲线图失败: {e}")

    def test(self, test_loader):
        """
        测试模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            dict: 测试结果
        """
        print("开始测试...")
        test_loss, test_acc, bitwise_sr, log2_sr, sample_sr, bit_match_pct = self.validate(test_loader)

        results = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_bitwise_sr": bitwise_sr,
            "test_log2_sr": log2_sr,
            "test_sample_sr": sample_sr,
            "test_bit_match_pct": bit_match_pct,
        }

        print(
            "测试结果: "
            f"test_loss: {test_loss:.4f}, "
            f"test_acc: {test_acc:.4f}, "
            f"test_bitwise_sr: {bitwise_sr:.4f}, "
            f"test_log2_sr: {log2_sr:.4f}, "
            f"test_sample_sr: {sample_sr:.8f}, "
            f"test_bit_match_pct: {bit_match_pct:.4f}"
        )

        # [MOD] 切换为整体绘图：不再按批次遍历收集曲线
        self.model.eval()
        self._plot_test_curves(
            test_loss, test_acc, bitwise_sr, log2_sr, sample_sr, bit_match_pct
        )

        return results

    def train(self, train_loader, val_loader):
        """
        完整训练过程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            dict: 训练结果
        """
        print(f"开始训练，使用设备: {self.device}")

        for epoch in range(self.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc, bitwise_sr, log2_sr, sample_sr, bit_match_pct = self.validate(
                val_loader
            )

            # 更新训练结果（兼容绘图回调）
            self.training_results["train_losses"].append(train_loss)
            self.training_results["val_losses"].append(val_loss)
            self.training_results["train_accuracies"].append(train_acc)
            self.training_results["val_accuracies"].append(val_acc)
            self.training_results["bitwise_success_rates"].append(float(bitwise_sr))
            self.training_results["log2_success_rates"].append(float(log2_sr))
            self.training_results["sample_success_rates"].append(float(sample_sr))
            self.training_results["bit_match_percentages"].append(float(bit_match_pct))
            self.training_results["total_epochs"] = epoch + 1

            # 调用绘图回调
            if self.plotting_callback:
                try:
                    self.plotting_callback.on_epoch_end(epoch)
                except Exception as e:
                    print(f"⚠ 绘图回调出错: {e}")

            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, "
                    f"Bitwise SR: {bitwise_sr:.4f}, "
                    f"Sample SR: {sample_sr:.8f}, "
                    f"Bit Match %: {bit_match_pct:.4f}"
                )

        # 训练结束时调用绘图回调
        if self.plotting_callback:
            try:
                self.plotting_callback.on_train_end()
                print("✓ 训练曲线图已保存")
            except Exception as e:
                print(f"⚠ 最终绘图出错: {e}")

        return {
            "train_losses": self.training_results["train_losses"],
            "val_losses": self.training_results["val_losses"],
            "train_accs": self.training_results["train_accuracies"],
            "val_accs": self.training_results["val_accuracies"],
            "final_val_acc": val_acc,
            "final_bitwise_sr": bitwise_sr,
            "final_log2_sr": log2_sr,
            "final_sample_sr": sample_sr,
            "final_bit_match_pct": bit_match_pct,
        }

    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")

    def load_model(self, path, input_dim, output_dim):
        """加载模型"""
        if self.model is None:
            self._init_model(input_dim, output_dim)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已从 {path} 加载")


def main():
    """简单的使用示例"""
    print("=" * 60)
    print("简化版训练模型使用示例（含绘图功能）")
    print("=" * 60)

    # 初始化训练器（启用绘图功能）
    trainer = TrainModelSimplify(
        hidden_dim=128,
        num_layers=2,
        lr=0.001,
        batch_size=32,
        epochs=20,  # 实际使用时可以设置更多轮数
        enable_plotting=True,  # 启用绘图功能
    )

    print("训练器配置:")
    print(f"  - 设备: {trainer.device}")
    print(f"  - 隐藏层维度: {trainer.hidden_dim}")
    print(f"  - LSTM层数: {trainer.num_layers}")
    print(f"  - 学习率: {trainer.lr}")
    print(f"  - 优化器: {trainer.optimizer_name}")
    print(f"  - 批次大小: {trainer.batch_size}")
    print(f"  - 训练轮数: {trainer.epochs}")
    print(f"  - 绘图功能: {'启用' if trainer.enable_plotting else '禁用'}")

    # 加载数据
    data_dir = "./data"  # 根据实际路径调整
    cipher_name = "present"
    rounds = 4

    print("\n加载数据:")
    print(f"  - 数据目录: {data_dir}")
    print(f"  - 密码算法: {cipher_name}")
    print(f"  - 轮数: {rounds}")

    try:
        train_loader, val_loader, test_loader = trainer.load_data(
            data_dir=data_dir, cipher_name=cipher_name, rounds=rounds
        )

        print(f"  - 训练批次数: {len(train_loader)}")
        print(f"  - 验证批次数: {len(val_loader)}")
        print(f"  - 测试批次数: {len(test_loader)}")

        # 训练
        print("\n开始训练...")
        train_results = trainer.train(train_loader, val_loader)

        print("\n训练结果:")
        print(f"  - 最终验证准确率: {train_results['final_val_acc']:.4f}")
        print(f"  - 最终比特成功率: {train_results['final_bitwise_sr']:.4f}")
        print(f"  - 最终Log2成功率: {train_results['final_log2_sr']:.4f}")
        print(f"  - 最终样本成功率: {train_results['final_sample_sr']:.4f}")

        # 测试
        test_results = trainer.test(test_loader)

        print("\n测试结果:")
        print(f"  - 测试准确率: {test_results['test_acc']:.4f}")
        print(f"  - 测试比特成功率: {test_results['test_bitwise_sr']:.4f}")
        print(f"  - 测试Log2成功率: {test_results['test_log2_sr']:.4f}")
        print(f"  - 测试样本成功率: {test_results['test_sample_sr']:.4f}")

        # 保存模型
        model_path = f"simple_model_{cipher_name}_round{rounds}.pth"
        trainer.save_model(model_path)

        print(f"\n模型已保存到: {model_path}")
        if trainer.plotting_callback:
            print("训练曲线图已保存到 plots/ 目录")
        print("=" * 60)
        print("训练完成!")

        return train_results, test_results

    except Exception as e:
        print(f"错误: {e}")
        print("请确保数据文件存在于正确的路径中")
        return None, None


if __name__ == "__main__":
    main()
