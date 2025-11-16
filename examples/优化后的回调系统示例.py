#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的回调系统使用示例

展示如何使用新的回调系统，其中回调函数可以直接访问训练器的training_results，
避免复杂的参数传递。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callbacks.base_callback import Callback, CallbackManager
from callbacks.early_stopping import EarlyStoppingCallback
from callbacks.plotting import PlottingCallback
from typing import Dict, Any, Optional


class MockTrainer:
    """
    模拟训练器类
    
    用于演示回调系统的使用方式
    """
    
    def __init__(self):
        """
        初始化模拟训练器
        """
        self.training_results = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_acc': 0.0,
            'val_acc': 0.0,
            'learning_rate': 0.001,
            'gradient_norm': 0.0,
            'bitwise_success_rate': 0.0,
            'log2_success_rate': 0.0,
            'early_stopped': False,
            'best_epoch': 0
        }
    
    def simulate_training(self, epochs: int = 20):
        """
        模拟训练过程
        
        演示如何在训练循环中使用优化后的回调系统
        
        Args:
            epochs (int): 训练轮数
        """
        print("🚀 开始模拟训练过程...")
        
        # 初始化回调管理器并设置训练器引用
        callback_manager = CallbackManager()
        callback_manager.set_trainer(self)  # 关键步骤：设置训练器引用
        
        # 添加早停回调
        early_stopping = EarlyStoppingCallback(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            mode='min'
        )
        callback_manager.add_callback(early_stopping)
        
        # 添加绘图回调（只在训练结束时绘制）
        plotting = PlottingCallback(
            save_dir="./demo_plots",
            experiment_name="callback_demo"
        )
        callback_manager.add_callback(plotting)
        
        # 添加自定义回调
        custom_callback = CustomMetricsCallback()
        callback_manager.add_callback(custom_callback)
        
        # 训练开始
        callback_manager.on_train_begin()
        
        # 模拟训练循环
        for epoch in range(epochs):
            # 模拟训练指标的变化
            self._simulate_epoch_metrics(epoch)
            
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {self.training_results['train_loss']:.4f}")
            print(f"  Val Loss: {self.training_results['val_loss']:.4f}")
            print(f"  Train Acc: {self.training_results['train_acc']:.4f}")
            print(f"  Val Acc: {self.training_results['val_acc']:.4f}")
            
            # 调用回调函数 - 注意：不需要传递复杂的logs参数
            callback_results = callback_manager.on_epoch_end(epoch)
            
            # 检查早停
            if callback_results.get('early_stop', False):
                print(f"\n⏹️ 早停触发，在第 {epoch + 1} 轮停止训练")
                break
            
            print("")
        
        # 训练结束
        callback_manager.on_train_end()
        print("✅ 训练完成！")
    
    def _simulate_epoch_metrics(self, epoch: int):
        """
        模拟每个epoch的指标变化
        
        Args:
            epoch (int): 当前epoch数
        """
        import random
        import math
        
        # 模拟损失下降（带一些随机波动）
        base_train_loss = 1.0 * math.exp(-epoch * 0.1) + random.uniform(-0.05, 0.05)
        base_val_loss = 1.1 * math.exp(-epoch * 0.08) + random.uniform(-0.08, 0.08)
        
        # 模拟准确率提升
        base_train_acc = min(0.95, 0.5 + epoch * 0.02 + random.uniform(-0.02, 0.02))
        base_val_acc = min(0.92, 0.45 + epoch * 0.018 + random.uniform(-0.03, 0.03))
        
        # 更新training_results
        self.training_results.update({
            'train_loss': max(0.01, base_train_loss),
            'val_loss': max(0.01, base_val_loss),
            'train_acc': max(0.0, base_train_acc),
            'val_acc': max(0.0, base_val_acc),
            'learning_rate': 0.001 * (0.95 ** epoch),
            'gradient_norm': max(0.1, 2.0 - epoch * 0.08),
            'bitwise_success_rate': min(1.0, 0.3 + epoch * 0.03),
            'log2_success_rate': min(1.0, 0.2 + epoch * 0.025)
        })


class CustomMetricsCallback(Callback):
    """
    自定义指标回调函数示例
    
    展示如何创建自定义回调函数，直接访问训练器的数据
    """
    
    def __init__(self):
        """
        初始化自定义回调
        """
        super().__init__()
        self.best_val_acc = 0.0
        self.metrics_history = []
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练开始时的初始化
        
        Args:
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        print("📊 自定义指标追踪器已启动")
        self.best_val_acc = 0.0
        self.metrics_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        每个epoch结束时的指标分析
        
        Args:
            epoch (int): 当前epoch数
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        # 直接从训练器获取数据 - 这是新设计的核心优势
        training_results = self.get_training_results()
        if training_results is None:
            return
        
        current_val_acc = training_results.get('val_acc', 0.0)
        
        # 记录指标历史
        self.metrics_history.append({
            'epoch': epoch,
            'val_acc': current_val_acc,
            'train_loss': training_results.get('train_loss', 0.0),
            'val_loss': training_results.get('val_loss', 0.0)
        })
        
        # 检查是否有新的最佳验证准确率
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            print(f"  🎯 新的最佳验证准确率: {current_val_acc:.4f}")
        
        # 计算改进率
        if len(self.metrics_history) > 1:
            prev_val_acc = self.metrics_history[-2]['val_acc']
            improvement = current_val_acc - prev_val_acc
            if improvement > 0.01:
                print(f"  📈 验证准确率显著提升: +{improvement:.4f}")
            elif improvement < -0.01:
                print(f"  📉 验证准确率下降: {improvement:.4f}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        训练结束时的总结报告
        
        Args:
            logs (dict, optional): 额外的训练日志信息（可选）
        """
        print("\n📊 自定义指标追踪器总结报告:")
        print(f"  最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"  总训练轮数: {len(self.metrics_history)}")
        
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            print(f"  最终验证准确率: {final_metrics['val_acc']:.4f}")
            print(f"  最终训练损失: {final_metrics['train_loss']:.4f}")
            print(f"  最终验证损失: {final_metrics['val_loss']:.4f}")


def demonstrate_old_vs_new_approach():
    """
    对比展示旧方法和新方法的区别
    """
    print("\n" + "="*60)
    print("📋 旧方法 vs 新方法对比")
    print("="*60)
    
    print("\n🔴 旧方法的问题:")
    print("1. 需要手动构造复杂的callback_logs字典")
    print("2. 数据重复传递，容易出错")
    print("3. 回调函数参数冗长，难以维护")
    print("4. 训练器和回调函数耦合度高")
    
    print("\n🟢 新方法的优势:")
    print("1. 回调函数直接访问训练器的training_results")
    print("2. 简化参数传递，减少代码重复")
    print("3. 更清晰的职责分离")
    print("4. 更容易扩展和维护")
    
    print("\n💡 核心改进:")
    print("- 回调函数通过self.get_training_results()直接获取数据")
    print("- 训练器通过callback_manager.set_trainer(self)设置引用")
    print("- 不再需要构造复杂的logs字典")


def main():
    """
    主函数
    
    运行回调系统优化示例
    """
    print("🎯 优化后的回调系统演示")
    print("="*50)
    
    # 展示新旧方法对比
    demonstrate_old_vs_new_approach()
    
    # 运行训练模拟
    print("\n" + "="*50)
    print("🏃‍♂️ 开始训练模拟")
    print("="*50)
    
    trainer = MockTrainer()
    trainer.simulate_training(epochs=15)
    
    print("\n" + "="*50)
    print("✨ 演示完成！")
    print("="*50)
    print("\n📝 总结:")
    print("- 回调函数现在可以直接访问训练器的数据")
    print("- 不再需要复杂的参数传递")
    print("- 代码更简洁、更易维护")
    print("- 扩展性更好，易于添加新的回调功能")


if __name__ == "__main__":
    main()