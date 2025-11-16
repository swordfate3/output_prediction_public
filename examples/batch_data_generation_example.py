#!/usr/bin/env python3
"""
分批次数据生成功能使用示例

这个示例展示如何使用新的分批次数据生成功能，包括：
1. 分批次生成npy文件
2. 自动合并和清理
3. 断点续传功能
4. 异常处理
"""

import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from utils.data_generator import generate_dataset  # noqa: E402
from utils.config import config  # noqa: E402
from ciphers import SmallPRESENT4  # noqa: E402
from utils.logger import getGlobalLogger  # noqa: E402

logger = getGlobalLogger()


def example_basic_batch_generation():
    """示例1: 基本的分批次生成"""
    logger.info("示例1: 基本的分批次生成")
    
    cipher = SmallPRESENT4(rounds=4)
    save_dir = os.path.join(config.DATA_DIR, "example_batch_basic")
    
    try:
        generate_dataset(
            cipher=cipher,
            num_keys=10,           # 10个密钥
            total_data=5000,       # 总样本5000，平均每密钥500，余数给最后一个密钥
            save_dir=save_dir,
            target_index=0,        # 提取第0位作为标签
            shuffle=True,          # 打乱数据
            batch_size=1000,       # 每批次1000个样本，将生成5个批次
            resume=True            # 启用断点续传
        )
        
        logger.info("✓ 基本分批次生成完成")
        
    except Exception as e:
        logger.error(f"基本分批次生成失败: {e}")


def example_large_dataset_generation():
    """示例2: 大数据集生成（内存优化）"""
    logger.info("示例2: 大数据集生成（内存优化）")
    
    cipher = SmallPRESENT4(rounds=4)
    save_dir = os.path.join(config.DATA_DIR, "example_large_dataset")
    
    try:
        generate_dataset(
            cipher=cipher,
            num_keys=100,          # 100个密钥
            total_data=100000,     # 总样本100,000，平均每密钥1000
            save_dir=save_dir,
            target_index=[0, 1, 2, 3],  # 提取多个比特位作为标签
            shuffle=True,
            batch_size=5000,       # 每批次5000个样本，减少内存使用
            resume=True
        )
        
        logger.info("✓ 大数据集生成完成")
        
    except Exception as e:
        logger.error(f"大数据集生成失败: {e}")


def example_resume_after_interruption():
    """示例3: 模拟中断后的断点续传"""
    logger.info("示例3: 模拟中断后的断点续传")
    
    cipher = SmallPRESENT4(rounds=4)
    save_dir = os.path.join(config.DATA_DIR, "example_resume")
    
    try:
        # 第一次运行（模拟中断）
        logger.info("第一次运行（模拟可能的中断）...")
        
        try:
            generate_dataset(
                cipher=cipher,
                num_keys=5,
                total_data=1000,       # 总样本1000，平均每密钥200
                save_dir=save_dir,
                target_index=0,
                shuffle=False,         # 不打乱，便于验证断点续传
                batch_size=300,        # 每批次300个样本，将生成4个批次
                resume=False           # 第一次不启用断点续传
            )
            
        except KeyboardInterrupt:
            logger.info("模拟用户中断...")
        
        # 第二次运行（断点续传）
        logger.info("第二次运行（启用断点续传）...")
        
        generate_dataset(
            cipher=cipher,
            num_keys=5,
            total_data=1000,
            save_dir=save_dir,
            target_index=0,
            shuffle=False,
            batch_size=300,
            resume=True            # 启用断点续传
        )
        
        logger.info("✓ 断点续传示例完成")
        
    except Exception as e:
        logger.error(f"断点续传示例失败: {e}")


def example_custom_batch_size():
    """示例4: 自定义批次大小以优化性能"""
    logger.info("示例4: 自定义批次大小以优化性能")
    
    cipher = SmallPRESENT4(rounds=4)
    
    # 不同的批次大小配置
    configs = [
        {"batch_size": 500, "desc": "小批次（适合内存受限环境）"},
        {"batch_size": 2000, "desc": "中等批次（平衡性能和内存）"},
        {"batch_size": 5000, "desc": "大批次（适合内存充足环境）"}
    ]
    
    for i, config_item in enumerate(configs, 1):
        save_dir = os.path.join(
            config.DATA_DIR, 
            f"example_batch_size_{config_item['batch_size']}"
        )
        
        try:
            logger.info(f"配置 {i}: {config_item['desc']}")
            
            generate_dataset(
                cipher=cipher,
                num_keys=5,
                total_data=500,        # 总样本500
                save_dir=save_dir,
                target_index=0,
                shuffle=True,
                batch_size=config_item['batch_size'],
                resume=True
            )
            
            logger.info(f"✓ 配置 {i} 完成")
            
        except Exception as e:
            logger.error(f"配置 {i} 失败: {e}")


def example_error_handling():
    """示例5: 错误处理和异常恢复"""
    logger.info("示例5: 错误处理和异常恢复")
    
    cipher = SmallPRESENT4(rounds=4)
    save_dir = os.path.join(config.DATA_DIR, "example_error_handling")
    
    try:
        # 模拟可能的错误情况
        generate_dataset(
            cipher=cipher,
            num_keys=3,
            total_data=300,
            save_dir=save_dir,
            target_index=0,
            shuffle=True,
            batch_size=100,
            resume=True
        )
        
        logger.info("✓ 错误处理示例完成")
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作，中间文件已保留用于断点续传")
        
    except Exception as e:
        logger.error(f"生成过程中出现错误: {e}")
        logger.info("中间文件已保留，可以使用断点续传功能恢复")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("分批次数据生成功能使用示例")
    logger.info("=" * 60)
    
    # 运行所有示例
    examples = [
        example_basic_batch_generation,
        example_large_dataset_generation,
        example_resume_after_interruption,
        example_custom_batch_size,
        example_error_handling
    ]
    
    for example_func in examples:
        try:
            example_func()
            logger.info("")  # 添加空行分隔
        except Exception as e:
            logger.error(f"示例 {example_func.__name__} 执行失败: {e}")
            logger.info("")
    
    logger.info("=" * 60)
    logger.info("所有示例执行完成")
    logger.info("=" * 60)
    
    # 使用说明
    logger.info("\n使用说明:")
    logger.info("1. batch_size: 控制每个批次的样本数量，影响内存使用")
    logger.info("2. resume: 启用断点续传，程序中断后可以继续生成")
    logger.info("3. shuffle: 控制是否打乱数据顺序")
    logger.info("4. 中间批次文件保存在 save_dir/batches/ 目录下")
    logger.info("5. 最终文件为 plain_texts.npy 和 cipher_texts.npy")
    logger.info("6. 生成完成后会自动清理中间文件")