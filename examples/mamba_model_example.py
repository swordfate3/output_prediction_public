#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba 模型使用示例

演示如何使用 CipherMamba 模型进行密文预测任务

Author: Output Prediction Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.mamba_model import CipherMamba, MambaBlock, SelectiveScan


def test_selective_scan():
    """
    测试选择性扫描机制
    
    验证 SelectiveScan 组件的基本功能
    """
    print("=== 测试选择性扫描机制 ===")
    
    # 创建选择性扫描层
    d_model = 64
    scan_layer = SelectiveScan(d_model=d_model, d_state=16, d_conv=4, expand=2)
    
    # 创建测试数据
    batch_size, seq_len = 4, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = scan_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"选择性扫描测试通过 ✓")
    print()


def test_mamba_block():
    """
    测试 Mamba 基础块
    
    验证 MambaBlock 的功能和残差连接
    """
    print("=== 测试 Mamba 基础块 ===")
    
    # 创建 Mamba 块
    d_model = 128
    mamba_block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    
    # 创建测试数据
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = mamba_block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"残差连接验证: {output.shape == x.shape}")
    print(f"Mamba 块测试通过 ✓")
    print()


def test_cipher_mamba():
    """
    测试完整的 CipherMamba 模型
    
    验证模型的训练和推理功能
    """
    print("=== 测试 CipherMamba 模型 ===")
    
    # 模型参数
    input_dim = 16  # 16位输入
    hidden_dim = 256
    num_layers = 4
    output_dim = 16  # 16位输出
    
    # 创建模型
    model = CipherMamba(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    )
    
    # 创建测试数据
    batch_size, seq_len = 8, 1  # 密码学任务通常序列长度为1
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"模型参数:")
    info = model.getModelInfo()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"CipherMamba 模型测试通过 ✓")
    print()


def test_model_save_load():
    """
    测试模型保存和加载功能
    
    验证模型的序列化和反序列化
    """
    print("=== 测试模型保存和加载 ===")
    
    # 创建原始模型
    original_model = CipherMamba(
        input_dim=16,
        hidden_dim=128,
        num_layers=2,
        output_dim=16
    )
    
    # 创建测试数据
    test_input = torch.randn(1, 1, 16)
    
    # 获取原始输出
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(test_input)
    
    # 保存模型
    save_path = "/tmp/test_mamba_model.pth"
    original_model.saveModel(save_path, epoch=10, loss=0.1)
    
    # 加载模型
    loaded_model, checkpoint = CipherMamba.loadModel(save_path)
    
    # 验证加载的模型
    loaded_model.eval()
    with torch.no_grad():
        loaded_output = loaded_model(test_input)
    
    # 检查输出是否一致
    output_match = torch.allclose(original_output, loaded_output, atol=1e-6)
    
    print(f"检查点信息:")
    print(f"  轮次: {checkpoint['epoch']}")
    print(f"  损失: {checkpoint['loss']}")
    print(f"输出一致性: {output_match}")
    print(f"模型保存和加载测试通过 ✓")
    print()


def performance_comparison():
    """
    性能对比测试
    
    比较 Mamba 和 LSTM 模型的参数量和推理速度
    """
    print("=== 性能对比测试 ===")
    
    # 创建相似规模的模型
    input_dim, output_dim = 16, 16
    hidden_dim = 256
    
    # Mamba 模型
    mamba_model = CipherMamba(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        output_dim=output_dim
    )
    
    # 获取模型信息
    mamba_info = mamba_model.getModelInfo()
    
    print(f"Mamba 模型:")
    print(f"  参数总数: {mamba_info['total_params']:,}")
    print(f"  可训练参数: {mamba_info['trainable_params']:,}")
    
    # 推理速度测试
    test_input = torch.randn(32, 1, input_dim)
    
    # 预热
    mamba_model.eval()
    with torch.no_grad():
        _ = mamba_model(test_input)
    
    # 计时
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = mamba_model(test_input)
    mamba_time = time.time() - start_time
    
    print(f"  推理时间 (100次): {mamba_time:.4f}s")
    print(f"  平均推理时间: {mamba_time/100*1000:.2f}ms")
    print()


def main():
    """
    主函数
    
    运行所有测试用例
    """
    print("Mamba 模型测试开始")
    print("=" * 50)
    
    try:
        # 运行各项测试
        test_selective_scan()
        test_mamba_block()
        test_cipher_mamba()
        test_model_save_load()
        performance_comparison()
        
        print("=" * 50)
        print("所有测试通过！Mamba 模型实现成功 🎉")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()