#!/usr/bin/env python3
"""
统一配置系统使用示例

本文件展示如何使用新的统一配置系统来进行实验配置、密码算法初始化和超参数管理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from models.train_model import TrainModel
from utils.data_generator import generate_dataset

def example_1_basic_usage():
    """
    示例1: 基本使用方法
    
    展示如何获取密码算法实例、最优超参数和实验配置
    """
    print("=== 示例1: 基本使用方法 ===")
    
    # 1. 获取密码算法实例
    cipher_name = "present"
    rounds = 4
    cipher = config.get_cipher_instance(cipher_name, rounds)
    print(f"密码算法实例: {type(cipher).__name__}")
    
    # 2. 获取最优超参数
    best_params = config.get_best_hyperparameters(cipher_name)
    print(f"最优超参数: {best_params}")
    
    # 3. 获取实验配置
    exp_config = config.get_experiment_config("exp1")
    print(f"实验1配置: {exp_config}")
    
    # 4. 获取完整配置
    complete_config = config.get_complete_config("exp1", cipher_name, rounds)
    print(f"完整配置包含: {list(complete_config.keys())}")

def example_2_experiment_setup():
    """
    示例2: 实验设置
    
    展示如何使用统一配置系统设置完整的实验环境
    """
    print("\n=== 示例2: 实验设置 ===")
    
    experiment_name = "exp2"
    cipher_name = "present"
    rounds = 4
    
    # 1. 验证配置
    try:
        config.validate_experiment_config(experiment_name, cipher_name, rounds)
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置验证失败: {e}")
        return
    
    # 2. 获取完整配置
    complete_config = config.get_complete_config(experiment_name, cipher_name, rounds)
    
    # 3. 初始化密码算法
    cipher = config.get_cipher_instance(cipher_name, rounds)
    print(f"✓ 密码算法初始化: {type(cipher).__name__}")
    
    # 4. 获取超参数
    hyperparams = complete_config["hyperparameters"]
    print(f"✓ 超参数配置: {hyperparams}")
    
    # 5. 获取数据路径
    data_paths = complete_config["data_paths"]
    print(f"✓ 数据路径: {data_paths}")

def example_3_hyperparameter_validation():
    """
    示例3: 超参数验证
    
    展示如何验证自定义超参数配置
    """
    print("\n=== 示例3: 超参数验证 ===")
    
    # 测试有效的超参数
    valid_params = {
        "hidden_dim": 300,
        "num_layers": 4,
        "lr": 0.01,
        "optimizer": "Adam",
        "batch_size": 250
    }
    
    try:
        config.validate_hyperparameters(valid_params)
        print("✓ 有效超参数验证通过")
    except ValueError as e:
        print(f"✗ 超参数验证失败: {e}")
    
    # 测试无效的超参数
    invalid_params = {
        "hidden_dim": 999,  # 不在有效范围内
        "num_layers": 10,   # 超出最大值
        "lr": 0.1,          # 不在有效范围内
    }
    
    try:
        config.validate_hyperparameters(invalid_params)
        print("✗ 无效超参数验证应该失败")
    except ValueError as e:
        print(f"✓ 无效超参数正确被拒绝: {e}")

def example_4_training_with_config():
    """
    示例4: 使用配置进行训练
    
    展示如何使用统一配置系统进行模型训练
    """
    print("\n=== 示例4: 使用配置进行训练 ===")
    
    experiment_name = "exp1"
    cipher_name = "present"
    rounds = 4
    
    # 1. 获取完整配置
    complete_config = config.get_complete_config(experiment_name, cipher_name, rounds)
    
    # 2. 提取配置参数
    hyperparams = complete_config["hyperparameters"]
    exp_config = complete_config["experiment"]
    
    # 3. 初始化训练器（使用统一配置的超参数）
    trainer = TrainModel(
        hidden_dim=hyperparams["hidden_dim"],
        num_layers=hyperparams["num_layers"],
        lr=hyperparams["lr"],
        optimizer=hyperparams["optimizer"],
        batch_size=hyperparams["batch_size"]
    )
    
    print(f"✓ 训练器初始化完成")
    print(f"  - 隐藏层维度: {hyperparams['hidden_dim']}")
    print(f"  - 网络层数: {hyperparams['num_layers']}")
    print(f"  - 学习率: {hyperparams['lr']}")
    print(f"  - 优化器: {hyperparams['optimizer']}")
    print(f"  - 批次大小: {hyperparams['batch_size']}")

def example_5_cipher_registry():
    """
    示例5: 密码算法注册表使用
    
    展示如何使用密码算法注册表功能
    """
    print("\n=== 示例5: 密码算法注册表使用 ===")
    
    # 1. 获取所有支持的密码算法
    supported_ciphers = config.get_supported_ciphers()
    print(f"支持的密码算法: {supported_ciphers}")
    
    # 2. 批量初始化所有密码算法
    print("\n批量初始化密码算法:")
    for cipher_name in supported_ciphers:
        try:
            cipher = config.get_cipher_instance(cipher_name, 4)
            print(f"✓ {cipher_name}: {type(cipher).__name__}")
        except Exception as e:
            print(f"✗ {cipher_name}: {e}")
    
    # 3. 验证密码算法名称
    print(f"\n验证 'present': {config.validate_cipher_name('present')}")
    print(f"验证 'invalid': {config.validate_cipher_name('invalid')}")

def example_6_data_path_generation():
    """
    示例6: 数据路径生成
    
    展示如何使用配置系统生成标准化的数据路径
    """
    print("\n=== 示例6: 数据路径生成 ===")
    
    experiments = ["exp1", "exp2", "exp3"]
    ciphers = ["present", "aes", "twine"]
    data_types = ["train", "test"]
    
    print("生成的数据路径:")
    for exp in experiments:
        for cipher in ciphers:
            for data_type in data_types:
                try:
                    path = config.get_data_path(exp, cipher, data_type)
                    print(f"  {exp}_{data_type}_{cipher}: {path}")
                except Exception as e:
                    print(f"  {exp}_{data_type}_{cipher}: 错误 - {e}")

if __name__ == "__main__":
    """主函数：运行所有示例"""
    print("统一配置系统使用示例")
    print("=" * 50)
    
    try:
        example_1_basic_usage()
        example_2_experiment_setup()
        example_3_hyperparameter_validation()
        example_4_training_with_config()
        example_5_cipher_registry()
        example_6_data_path_generation()
        
        print("\n" + "=" * 50)
        print("✓ 所有示例运行完成！")
        
    except Exception as e:
        print(f"\n✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()