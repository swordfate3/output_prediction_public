#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回调管理器使用示例

演示如何使用CallbackManager来管理多个回调函数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callbacks.base_callback import CallbackManager, Callback
from callbacks.early_stopping import EarlyStoppingCallback
from callbacks.plotting import PlottingCallback
from typing import Dict, Any, Optional


class CustomLoggerCallback(Callback):
    """
    自定义日志回调函数示例
    
    演示如何创建自定义回调函数
    """
    
    def __init__(self, log_frequency: int = 1):
        """
        初始化自定义日志回调
        
        Args:
            log_frequency (int): 日志记录频率（每多少个epoch记录一次）
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.epoch_count = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练开始时的日志记录
        
        Args:
            logs (dict, optional): 训练日志信息
        """
        print("🚀 训练开始！")
        print(f"📊 日志记录频率: 每 {self.log_frequency} 个epoch")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        每个epoch结束时的日志记录
        
        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 训练日志信息
        """
        self.epoch_count += 1
        
        if (epoch + 1) % self.log_frequency == 0:
            if logs:
                train_loss = logs.get('train_loss', 'N/A')
                val_loss = logs.get('val_loss', 'N/A')
                train_acc = logs.get('train_acc', 'N/A')
                val_acc = logs.get('val_acc', 'N/A')
                
                train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
                val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
                train_acc_str = f"{train_acc:.4f}" if isinstance(train_acc, (int, float)) else str(train_acc)
                val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else str(val_acc)
                
                print(f"📈 Epoch {epoch + 1}: "
                      f"Train Loss: {train_loss_str}, "
                      f"Val Loss: {val_loss_str}, "
                      f"Train Acc: {train_acc_str}, "
                      f"Val Acc: {val_acc_str}")
        
        return {}
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练结束时的日志记录
        
        Args:
            logs (dict, optional): 训练日志信息
        """
        print(f"🎉 训练完成！总共进行了 {self.epoch_count} 个epoch")
        if logs:
            final_train_loss = logs.get('train_loss', 'N/A')
            final_val_loss = logs.get('val_loss', 'N/A')
            print(f"📊 最终结果: Train Loss: {final_train_loss}, Val Loss: {final_val_loss}")


def demonstrate_callback_manager():
    """
    演示CallbackManager的使用方法
    
    展示如何创建、添加、管理和使用多个回调函数
    """
    print("=" * 60)
    print("🔧 CallbackManager 使用示例")
    print("=" * 60)
    
    # 1. 创建回调管理器
    print("\n1️⃣ 创建回调管理器")
    callback_manager = CallbackManager()
    print(f"✅ 回调管理器已创建，当前回调数量: {callback_manager.get_callback_count()}")
    
    # 2. 创建并添加各种回调函数
    print("\n2️⃣ 添加回调函数")
    
    # 添加自定义日志回调
    logger_callback = CustomLoggerCallback(log_frequency=2)
    callback_manager.add_callback(logger_callback)
    print("✅ 已添加自定义日志回调")
    
    # 添加早停回调
    early_stopping = EarlyStoppingCallback(
        monitor='val_loss',
        patience=3,
        min_delta=0.001,
        mode='min'
    )
    callback_manager.add_callback(early_stopping)
    print("✅ 已添加早停回调")
    
    # 添加绘图回调
    plotting_callback = PlottingCallback(
        save_dir="./example_plots",
        experiment_name="callback_demo",
        plot_frequency=5
    )
    callback_manager.add_callback(plotting_callback)
    print("✅ 已添加绘图回调")
    
    print(f"\n📊 当前注册的回调函数: {callback_manager.get_callback_names()}")
    print(f"📊 回调函数总数: {callback_manager.get_callback_count()}")
    
    # 3. 模拟训练过程
    print("\n3️⃣ 模拟训练过程")
    
    # 训练开始
    callback_manager.on_train_begin()
    
    # 模拟训练循环
    for epoch in range(10):
        # 模拟训练指标
        train_loss = 1.0 - epoch * 0.08 + (epoch % 3) * 0.02
        val_loss = 1.1 - epoch * 0.07 + (epoch % 2) * 0.03
        train_acc = 0.5 + epoch * 0.04
        val_acc = 0.45 + epoch * 0.035
        
        # 准备日志数据
        logs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': 0.001 * (0.95 ** epoch),
            'gradient_norm': 1.5 - epoch * 0.1,
            'training_results': {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'stop_training': False
            }
        }
        
        # 调用epoch结束回调
        callback_results = callback_manager.on_epoch_end(epoch, logs)
        
        # 检查是否需要早停
        if callback_results.get('stop_training', False):
            print(f"⏹️ 早停触发，在第 {epoch + 1} 个epoch停止训练")
            break
    
    # 训练结束
    final_logs = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    callback_manager.on_train_end(final_logs)
    
    # 4. 演示回调管理功能
    print("\n4️⃣ 回调管理功能演示")
    
    # 移除特定回调
    print(f"\n移除前回调数量: {callback_manager.get_callback_count()}")
    callback_manager.remove_callback(logger_callback)
    print(f"移除自定义日志回调后数量: {callback_manager.get_callback_count()}")
    print(f"剩余回调: {callback_manager.get_callback_names()}")
    
    # 清空所有回调
    callback_manager.clear_callbacks()
    print(f"\n清空所有回调后数量: {callback_manager.get_callback_count()}")
    
    print("\n=" * 60)
    print("✨ CallbackManager 演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_callback_manager()