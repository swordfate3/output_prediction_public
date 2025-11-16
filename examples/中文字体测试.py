#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体显示测试

验证PlottingCallback的中文字体配置是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callbacks.plotting import PlottingCallback
import numpy as np


class FontTestTrainer:
    """简单的字体测试训练器"""
    
    def __init__(self):
        self.callbacks = []
        self.current_epoch = 0
        
    def add_callback(self, callback):
        callback.set_trainer(self)
        self.callbacks.append(callback)
        
    def get_training_results(self):
        # 生成模拟的训练结果
        return {
            'train_loss': 0.5 - 0.05 * self.current_epoch + np.random.normal(0, 0.02),
            'val_loss': 0.6 - 0.04 * self.current_epoch + np.random.normal(0, 0.03),
            'train_acc': 0.5 + 0.08 * self.current_epoch + np.random.normal(0, 0.01),
            'val_acc': 0.4 + 0.07 * self.current_epoch + np.random.normal(0, 0.02)
        }
        
    def train(self, epochs=3):
        print(f"开始训练，共{epochs}个轮次...")
        
        for callback in self.callbacks:
            callback.on_train_begin()
            
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"轮次 {epoch + 1}/{epochs}")
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch)
                
        for callback in self.callbacks:
            callback.on_train_end()
            
        print("训练完成！")


def test_chinese_font():
    """测试中文字体显示"""
    print("=" * 50)
    print("中文字体显示测试")
    print("=" * 50)
    
    # 测试中文标签
    print("\n测试1: 标准中文标签")
    print("-" * 30)
    
    zh_callback = PlottingCallback(
        save_dir="demo_plots",
        experiment_name="font_test_zh",
        language="zh"
    )
    
    print(f"损失标题: {zh_callback.get_available_labels()['loss_title']}")
    print(f"准确率标题: {zh_callback.get_available_labels()['accuracy_title']}")
    print(f"训练损失: {zh_callback.get_available_labels()['train_loss']}")
    
    trainer = FontTestTrainer()
    trainer.add_callback(zh_callback)
    trainer.train(epochs=3)
    
    print("\n测试2: 复杂中文标签")
    print("-" * 30)
    
    complex_labels = {
        'loss_title': '深度神经网络损失函数收敛曲线',
        'accuracy_title': '模型分类准确率性能指标',
        'train_loss': '训练集损失值',
        'val_loss': '验证集损失值',
        'train_acc': '训练集准确率',
        'val_acc': '验证集准确率',
        'training_curves': '人工智能模型训练监控面板',
        'final': '最终结果'
    }
    
    complex_callback = PlottingCallback(
        save_dir="demo_plots",
        experiment_name="font_test_complex",
        language="zh",
        custom_labels=complex_labels
    )
    
    print(f"复杂标题: {complex_callback.get_available_labels()['loss_title']}")
    
    trainer2 = FontTestTrainer()
    trainer2.add_callback(complex_callback)
    trainer2.train(epochs=3)
    
    print("\n=" * 50)
    print("字体测试完成！")
    print("=" * 50)
    
    print("\n生成的测试图片:")
    plot_dir = "./demo_plots"
    if os.path.exists(plot_dir):
        files = [f for f in os.listdir(plot_dir) if f.startswith('font_test') and f.endswith('.png')]
        for file in sorted(files):
            print(f"  {file}")
    
    print("\n说明:")
    print("- 如果图片中的中文显示正常，说明字体配置成功")
    print("- 如果出现方框或乱码，说明需要安装中文字体")
    print("- Linux系统推荐安装: sudo apt-get install fonts-wqy-microhei")


if __name__ == "__main__":
    test_chinese_font()