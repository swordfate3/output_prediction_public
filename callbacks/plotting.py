#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图回调函数

实现训练过程中的结果可视化功能，包括训练曲线绘制、指标监控等
"""

import os
from typing import Dict, Optional
from .base_callback import Callback
from utils.directory_manager import get_plots_directory, ensure_directory
from utils.logger import getGlobalLogger
import time

logger = getGlobalLogger()


class PlottingCallback(Callback):
    """
    绘图回调函数

    负责绘制和保存训练过程中的各种图表
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        experiment_name: str = "training",
        cipher_name: str = "cipher",
        rounds: int = 4,
        save_format: str = "png",
        dpi: int = 300,
        figsize: tuple = (12, 10),
        custom_labels: Optional[Dict[str, str]] = None,
    ):
        """
        初始化绘图回调函数

        Args:
            save_dir (str, optional): 图片保存目录，如果为None则使用目录管理器的默认图表目录
            experiment_name (str): 实验名称
            cipher_name (str): 加密算法名称
            save_format (str): 图片保存格式
            dpi (int): 图片分辨率
            figsize (tuple): 图片尺寸
            custom_labels (dict, optional): 自定义标签字典，用于覆盖默认标签

        """
        super().__init__()

        # 使用目录管理器获取图表目录
        if save_dir is None:
            self.save_dir = get_plots_directory(create=True)
            logger.info(f"使用默认图表目录: {self.save_dir}")
        else:
            self.save_dir = ensure_directory(save_dir)
            logger.info(f"使用指定图表目录: {self.save_dir}")
        self.experiment_name = experiment_name
        self.cipher_name = cipher_name
        self.rounds = rounds
        self.save_format = save_format
        self.dpi = dpi
        self.figsize = figsize
        self.custom_labels = custom_labels or {}

        logger.info(
            f"初始化绘图回调函数 - 实验名称: {experiment_name}, "
            f"格式: {save_format}, DPI: {dpi}"
        )

        # 设置默认标签
        self._setup_labels()

        # 检查matplotlib是否可用
        self.matplotlib_available = self._check_matplotlib()
        if self.matplotlib_available:
            logger.info("matplotlib检查通过，绘图功能已启用")
        else:
            logger.warning("matplotlib不可用，绘图功能已禁用")

        # 设置matplotlib基本参数
        if self.matplotlib_available:
            self._setup_matplotlib()

    def _setup_labels(self) -> None:
        """
        设置绘图标签

        设置英文标签文本
        """
        # 默认英文标签
        default_labels = {
            "title": f"{self.cipher_name} {self.rounds}-Round Training Curves",
            "epoch": "Epoch",
            "loss": "Loss",
            "accuracy": "Accuracy",
            "train_loss": "Training Loss",
            "val_loss": "Validation Loss",
            "train_acc": "Training Accuracy",
            "val_acc": "Validation Accuracy",
            "best_epoch": "Best Epoch",
            "early_stop": "Early Stop",
            "final_train_loss": "Final Training Loss",
            "final_val_loss": "Final Validation Loss",
            "final_train_acc": "Final Training Accuracy",
            "final_val_acc": "Final Validation Accuracy",
            "loss_subplot": "Loss Curves",
            "accuracy_subplot": "Accuracy Curves"
        }

        # 合并自定义标签
        self.labels = {**default_labels, **self.custom_labels}

    def _setup_matplotlib(self) -> None:
        """
        设置matplotlib基本参数

        设置matplotlib的基本显示参数
        """
        if not self.matplotlib_available:
            return

        try:
            import matplotlib.pyplot as plt

            # 设置基本字体和显示参数
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = True
            plt.rcParams["figure.figsize"] = self.figsize
            plt.rcParams["savefig.dpi"] = self.dpi
            plt.rcParams["savefig.bbox"] = "tight"

            logger.debug("matplotlib基本参数设置完成")

        except ImportError:
            logger.warning("matplotlib导入失败，跳过参数设置")
        except Exception as e:
            logger.error(f"matplotlib参数设置过程中发生错误: {e}")

    def _check_matplotlib(self) -> bool:
        """
        检查matplotlib是否可用

        Returns:
            bool: matplotlib是否可用
        """
        try:
            import matplotlib

            # 设置非交互式后端
            matplotlib.use("Agg")
            return True
        except ImportError:
            logger.warning("matplotlib未安装，绘图功能将被禁用")
            return False
        except Exception as e:
            logger.warning(f"matplotlib检查失败: {e}")
            return False

    def on_epoch_end(self, epoch, logs=None):
        """
        每个epoch结束时的回调处理

        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 额外的训练日志信息（已弃用，保留兼容性）
        """
        if not self.matplotlib_available:
            return

        # 验证是否能获取训练结果数据
        training_results = self.get_training_results()
        if training_results is None:
            logger.warning("无法获取训练结果数据，跳过绘图")
            return

        # 在每个epoch结束时，数据已经由TrainModel维护，这里不需要额外处理
        # logger.debug(f"Epoch {epoch} 绘图回调处理完成")

    def on_train_end(self, logs=None):
        """
        训练结束时生成最终图表

        Args:
            logs (dict, optional): 额外的训练日志信息（已弃用，保留兼容性）
        """
        if not self.matplotlib_available:
            return

        try:
            # 从训练器获取训练结果数据
            training_results = self.get_training_results()
            if training_results is None:
                logger.warning("无法获取训练结果数据，跳过最终绘图")
                return

            # 获取最终epoch数
            final_epoch = training_results.get("total_epochs", 0)
            self._plot_training_curves(final_epoch, final=True)
            logger.info("最终训练曲线已生成")
        except Exception as e:
            logger.error(f"生成最终训练曲线时出错: {e}")

    def _plot_training_curves(self, current_epoch: int,
                              final: bool = False) -> None:
        """
        绘制训练曲线

        Args:
            current_epoch (int): 当前epoch数
            final (bool): 是否为最终绘制
        """
        if not self.matplotlib_available:
            return

        try:
            import matplotlib.pyplot as plt

            # 确保使用基本字体设置
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = True

            # 从训练器获取训练结果数据
            training_results = self.get_training_results()
            if training_results is None:
                logger.warning("无法获取训练结果数据，跳过绘图")
                return

            # 直接从training_results获取历史数据列表
            train_losses = training_results.get("train_losses", [])
            val_losses = training_results.get("val_losses", [])

            if not train_losses and not val_losses:
                logger.warning("训练历史数据为空，跳过绘图")
                return

            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle(self.labels["title"], fontsize=16,
                         fontweight="bold")

            # 获取epoch列表（基于最长的历史数据）
            max_epochs = max(len(train_losses), len(val_losses))
            epochs = list(range(1, max_epochs + 1))
            if not epochs:
                logger.warning("没有epoch数据，跳过绘图")
                return

            # 1. 损失曲线
            ax1 = axes[0, 0]

            if train_losses:
                ax1.plot(
                    epochs[:len(train_losses)],
                    train_losses,
                    "b-",
                    label=self.labels["train_loss"],
                    linewidth=2,
                )

            if val_losses:
                ax1.plot(
                    epochs[:len(val_losses)],
                    val_losses,
                    "r-",
                    label=self.labels["val_loss"],
                    linewidth=2,
                )

            ax1.set_xlabel(self.labels["epoch"])
            ax1.set_ylabel(self.labels["loss"])
            ax1.set_title(self.labels["loss_subplot"])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 准确率曲线
            ax2 = axes[0, 1]
            train_accs = training_results.get("train_accuracies", [])
            val_accs = training_results.get("val_accuracies", [])

            if train_accs:
                ax2.plot(
                    epochs[:len(train_accs)],
                    train_accs,
                    "b-",
                    label=self.labels["train_acc"],
                    linewidth=2,
                )

            if val_accs:
                ax2.plot(
                    epochs[:len(val_accs)],
                    val_accs,
                    "r-",
                    label=self.labels["val_acc"],
                    linewidth=2,
                )

            ax2.set_xlabel(self.labels["epoch"])
            ax2.set_ylabel(self.labels["accuracy"])
            ax2.set_title(self.labels["accuracy_subplot"])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 样本完全匹配成功率曲线（替换学习率曲线）
            ax3 = axes[1, 0]
            sample_rates = training_results.get("sample_success_rates", [])

            if sample_rates:
                ax3.plot(
                    epochs[:len(sample_rates)],
                    sample_rates,
                    "g-",
                    label="Sample Success Rate",
                    linewidth=2,
                )
                ax3.set_xlabel(self.labels["epoch"])
                ax3.set_ylabel("Success Rate")
                ax3.set_title("Sample Success Rate Curve")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # 4. 成功率曲线
            ax4 = axes[1, 1]
            success_rate_plotted = False

            bitwise_rates = training_results.get("bitwise_success_rates", [])
            log2_rates = training_results.get("log2_success_rates", [])
            # bit_match_pcts = training_results.get("bit_match_percentages", [])

            # [MOD] 使用原始序列：只要序列非空即可绘制
            if bitwise_rates:
                ax4.plot(
                    epochs[:len(bitwise_rates)],
                    bitwise_rates,
                    "purple",
                    label="Bitwise Success Rate",
                    linewidth=2,
                )
                success_rate_plotted = True

            if log2_rates:
                ax4.plot(
                    epochs[:len(log2_rates)],
                    log2_rates,
                    "orange",
                    label="Log2 Success Rate",
                    linewidth=2,
                )
                success_rate_plotted = True

            # if bit_match_pcts:
            #     ax4.plot(
            #         epochs[:len(bit_match_pcts)],
            #         bit_match_pcts,
            #         "blue",
            #         label="Bit Match Percentage",
            #         linewidth=2,
            #     )
            #     success_rate_plotted = True

            if success_rate_plotted:
                ax4.set_xlabel(self.labels["epoch"])
                ax4.set_ylabel("Success Rate")
                ax4.set_title("Success Rate Curves")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                # 如果没有成功率数据，绘制梯度范数
                gradient_norms = training_results.get("gradient_norms", [])
                if gradient_norms:
                    ax4.plot(
                        epochs[:len(gradient_norms)],
                        gradient_norms,
                        "brown",
                        label="Gradient Norm",
                        linewidth=2,
                    )
                    ax4.set_xlabel(self.labels["epoch"])
                    ax4.set_ylabel("Gradient Norm")
                    ax4.set_title("Gradient Norm Curve")
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图片
            suffix = "_final" if final else f"_epoch_{current_epoch}"
            filename = (
                f"{self.experiment_name}_training_curves{suffix}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}."
                f"{self.save_format}"
            )
            filepath = os.path.join(self.save_dir, filename)

            plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight",
                        pad_inches=0.1)
            plt.close()

            if final:
                logger.info(f"Final training curves saved: {filepath}")
            else:
                logger.debug(f"训练曲线已保存: {filepath}")

        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")
            # 确保图形被关闭
            try:
                plt.close("all")
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"PlottingCallback(save_dir='{self.save_dir}', "
            f"experiment_name='{self.experiment_name}', "
            f"cipher_name='{self.cipher_name}', "
            f"rounds={self.rounds}, "
            f"save_format='{self.save_format}', "
            f"matplotlib_available={self.matplotlib_available})"
        )
