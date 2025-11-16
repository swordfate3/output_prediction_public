#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础 Mamba 模型测试

验证 Mamba 模型的核心功能，避免环境兼容性问题

Author: Output Prediction Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def testMambaImport():
    """
    测试 Mamba 模型导入
    
    验证模型类是否可以正确导入
    """
    print("=== 测试 Mamba 模型导入 ===")
    
    try:
        from models import CipherMamba, MambaBlock, SelectiveScan
        print("✓ CipherMamba 导入成功")
        print("✓ MambaBlock 导入成功") 
        print("✓ SelectiveScan 导入成功")
        print("Mamba 模型导入测试通过 ✓\n")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def testMambaBasicCreation():
    """
    测试 Mamba 模型基本创建
    
    验证模型实例化是否正常
    """
    print("=== 测试 Mamba 模型创建 ===")
    
    try:
        from models import CipherMamba
        
        # 创建模型
        model = CipherMamba(
            input_dim=16,
            hidden_dim=128,
            num_layers=2,
            output_dim=16,
            d_state=8,
            d_conv=4,
            expand=2,
            dropout=0.1
        )
        
        print(f"✓ 模型创建成功")
        print(f"✓ 模型类型: {type(model)}")
        
        # 获取模型信息
        info = model.getModelInfo()
        print(f"✓ 参数总数: {info['total_params']:,}")
        print(f"✓ 模型大小: {info['model_size_mb']:.2f} MB")
        
        print("Mamba 模型创建测试通过 ✓\n")
        return model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def testMambaForward(model):
    """
    测试 Mamba 模型前向传播
    
    验证模型的前向传播功能
    
    Args:
        model: Mamba 模型实例
    """
    print("=== 测试 Mamba 前向传播 ===")
    
    if model is None:
        print("✗ 模型为空，跳过前向传播测试")
        return False
    
    try:
        # 创建测试数据
        batch_size, seq_len, input_dim = 4, 1, 16
        x = torch.randn(batch_size, seq_len, input_dim)
        
        print(f"✓ 输入形状: {x.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出形状
        expected_shape = (batch_size, seq_len, 16)
        if output.shape == expected_shape:
            print(f"✓ 输出形状正确: {output.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {output.shape}")
            return False
        
        print("Mamba 前向传播测试通过 ✓\n")
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def testMambaSaveLoad(model):
    """
    测试 Mamba 模型保存和加载
    
    验证模型的序列化功能
    
    Args:
        model: Mamba 模型实例
    """
    print("=== 测试 Mamba 保存和加载 ===")
    
    if model is None:
        print("✗ 模型为空，跳过保存加载测试")
        return False
    
    try:
        from models import CipherMamba
        
        # 创建测试数据
        test_input = torch.randn(1, 1, 16)
        
        # 获取原始输出
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # 保存模型
        save_path = "/tmp/test_basic_mamba.pth"
        model.saveModel(save_path, epoch=5, loss=0.05)
        print(f"✓ 模型已保存到: {save_path}")
        
        # 加载模型
        loaded_model, checkpoint = CipherMamba.loadModel(save_path)
        print(f"✓ 模型已加载")
        print(f"✓ 检查点轮次: {checkpoint['epoch']}")
        print(f"✓ 检查点损失: {checkpoint['loss']}")
        
        # 验证加载的模型
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(test_input)
        
        # 检查输出一致性
        output_match = torch.allclose(original_output, loaded_output, atol=1e-6)
        print(f"✓ 输出一致性: {output_match}")
        
        if output_match:
            print("Mamba 保存和加载测试通过 ✓\n")
            return True
        else:
            print("✗ 输出不一致")
            return False
        
    except Exception as e:
        print(f"✗ 保存加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数
    
    运行所有基础测试
    """
    print("基础 Mamba 模型测试开始")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 测试导入
    if testMambaImport():
        success_count += 1
    
    # 测试模型创建
    model = testMambaBasicCreation()
    if model is not None:
        success_count += 1
        
        # 测试前向传播
        if testMambaForward(model):
            success_count += 1
        
        # 测试保存加载
        if testMambaSaveLoad(model):
            success_count += 1
    
    # 总结
    print("=" * 50)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有基础测试通过！Mamba 模型实现成功！")
    else:
        print(f"⚠️  有 {total_tests - success_count} 个测试失败")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()