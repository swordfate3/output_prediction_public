#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言绘图回调函数示例

演示如何使用PlottingCallback的中英文标题和自定义标签功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from callbacks.plotting import PlottingCallback
from callbacks.base_callback import CallbackManager

# 简化的训练器类用于演示
class SimpleTrainer:
    def __init__(self, model, train_data, val_data, callback_manager=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.callback_manager = callback_manager or CallbackManager()
        self.training_results = {}
        
        # 设置回调管理器的训练器引用
        if self.callback_manager:
            self.callback_manager.set_trainer(self)
    
    def train(self, epochs=5, verbose=True):
        """简化的训练循环"""
        import random
        
        for epoch in range(epochs):
            # 模拟训练结果
            train_loss = 1.0 - (epoch * 0.15) + random.uniform(-0.1, 0.1)
            val_loss = 1.2 - (epoch * 0.12) + random.uniform(-0.1, 0.1)
            train_acc = 0.5 + (epoch * 0.08) + random.uniform(-0.05, 0.05)
            val_acc = 0.45 + (epoch * 0.09) + random.uniform(-0.05, 0.05)
            
            # 更新训练结果
            self.training_results = {
                'train_loss': max(0.1, train_loss),
                'val_loss': max(0.1, val_loss),
                'train_acc': min(0.95, max(0.1, train_acc)),
                'val_acc': min(0.95, max(0.1, val_acc))
            }
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {self.training_results['train_loss']:.4f}, "
                      f"Val Loss: {self.training_results['val_loss']:.4f}, "
                      f"Train Acc: {self.training_results['train_acc']:.4f}, "
                      f"Val Acc: {self.training_results['val_acc']:.4f}")
            
            # 调用回调函数
            if self.callback_manager:
                self.callback_manager.on_epoch_end(epoch)
        
        # 训练结束回调
        if self.callback_manager:
            self.callback_manager.on_train_end()


def create_simple_model():
    """
    创建简单的测试模型
    
    Returns:
        nn.Module: 简单的神经网络模型
    """
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )


def demo_multilingual_plotting():
    """
    演示多语言绘图功能
    
    展示中文、英文和自定义标签的使用
    """
    print("=" * 50)
    print("多语言绘图回调函数演示")
    print("=" * 50)
    
    # 准备模拟数据
    train_data = torch.randn(800, 64)  # 模拟训练数据
    val_data = torch.randn(200, 64)    # 模拟验证数据
    
    # 创建模型
    model = create_simple_model()
    
    print("\n测试1: 中文标题 (默认)")
    print("-" * 30)
    
    # 中文绘图回调
    plotting_zh = PlottingCallback(
        save_dir="./demo_plots",
        experiment_name="multilang_zh",
        language="zh"  # 中文
    )
    
    callback_manager = CallbackManager()
    callback_manager.add_callback(plotting_zh)
    
    trainer = SimpleTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        callback_manager=callback_manager
    )
    
    # 训练5个epoch
    trainer.train(epochs=5, verbose=False)
    
    print("中文图表已生成")
    
    print("\n测试2: 英文标题")
    print("-" * 30)
    
    # 重新创建模型
    model = create_simple_model()
    
    # 英文绘图回调
    plotting_en = PlottingCallback(
        save_dir="./demo_plots",
        experiment_name="multilang_en",
        language="en"  # 英文
    )
    
    callback_manager = CallbackManager()
    callback_manager.add_callback(plotting_en)
    
    trainer = SimpleTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        callback_manager=callback_manager
    )
    
    # 训练5个epoch
    trainer.train(epochs=5, verbose=False)
    
    print("英文图表已生成")
    
    print("\n测试3: 自定义标签")
    print("-" * 30)
    
    # 重新创建模型
    model = create_simple_model()
    
    # 自定义标签
    custom_labels = {
        'loss_title': '损失函数变化',
        'accuracy_title': '准确率提升',
        'train_loss': '训练集损失',
        'val_loss': '验证集损失',
        'train_acc': '训练集准确率',
        'val_acc': '验证集准确率',
        'training_curves': '深度学习训练监控',
        'final': '训练完成'
    }
    
    # 自定义标签绘图回调
    plotting_custom = PlottingCallback(
        save_dir="./demo_plots",
        experiment_name="multilang_custom",
        language="zh",
        custom_labels=custom_labels
    )
    
    callback_manager = CallbackManager()
    callback_manager.add_callback(plotting_custom)
    
    trainer = SimpleTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        callback_manager=callback_manager
    )
    
    # 训练5个epoch
    trainer.train(epochs=5, verbose=False)
    
    print("自定义标签图表已生成")
    
    print("\n测试4: 动态语言切换")
    print("-" * 30)
    
    # 演示动态语言切换
    plotting_dynamic = PlottingCallback(
        save_dir="./demo_plots",
        experiment_name="multilang_dynamic",
        language="zh"
    )
    
    print(f"初始语言: {plotting_dynamic.language}")
    print(f"当前标签: {list(plotting_dynamic.get_available_labels().keys())[:3]}...")
    
    # 切换到英文
    plotting_dynamic.set_language("en")
    print(f"切换后语言: {plotting_dynamic.language}")
    print(f"英文标签示例: {plotting_dynamic.get_available_labels()['loss_title']}")
    
    # 切换回中文并添加自定义标签
    plotting_dynamic.set_language("zh", {'loss_title': '自定义损失曲线'})
    print(f"最终语言: {plotting_dynamic.language}")
    print(f"自定义标签: {plotting_dynamic.get_available_labels()['loss_title']}")
    
    print("\n" + "=" * 50)
    print("多语言演示完成！")
    print("=" * 50)
    
    print("\n生成的图片文件:")
    plot_dir = "./demo_plots"
    if os.path.exists(plot_dir):
        files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        for file in sorted(files):
            print(f"  {file}")
    
    print("\n使用说明:")
    print("- language='zh': 使用中文标题和标签")
    print("- language='en': 使用英文标题和标签")
    print("- custom_labels: 自定义任意标签文本")
    print("- set_language(): 动态切换语言")
    print("- get_available_labels(): 查看当前所有标签")


if __name__ == "__main__":
    demo_multilingual_plotting()