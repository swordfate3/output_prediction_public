import numpy as np
from models.train_model import TrainModel
from utils.data_generator import generate_dataset
from utils.config import config
from typing import Dict, Any
import os
from utils.logger import getGlobalLogger

# 创建实验日志器
logger = getGlobalLogger()


def generate_data(cipher_name: str, rounds: int = 4):
    """
    生成exp2实验所需的训练和测试数据
    
    详细描述：为攻击成功率验证实验生成训练和测试数据集
    
    Args:
        cipher_name (str): 密码算法名称
        rounds (int): 密码轮数，默认为4
        
    Returns:
        None
        
    Example:
        >>> generate_data("present", 4)
    """
    # 初始化密码算法
    cipher = config.get_cipher_instance(cipher_name, rounds)
    # 实验2初始化配置
    experimental_2 = config.get_experiment_config("exp2")
    
    logger.info(f"开始生成exp2训练数据 - {cipher_name}, {rounds}轮")
    # 生成训练数据
    generate_dataset(
        cipher=cipher,
        num_keys=experimental_2["keys"],
        samples_per_key=experimental_2["train_samples"],
        save_dir=os.path.join(
            config.DATA_DIR,
            f"exp2_train_{cipher_name}_round{rounds}"
        ),
        target_index=experimental_2["target_index"],
        shuffle=True
    )
    
    logger.info(f"开始生成exp2测试数据 - {cipher_name}, {rounds}轮")
    # 生成测试数据
    generate_dataset(
        cipher=cipher,
        num_keys=experimental_2["keys"],
        samples_per_key=experimental_2["test_samples"],
        save_dir=os.path.join(
            config.DATA_DIR,
            f"exp2_test_{cipher_name}_round{rounds}"
        ),
        target_index=experimental_2["target_index"],
        shuffle=True
    )
    
    logger.info(f"exp2数据生成完成 - {cipher_name}, {rounds}轮")


def train_model(cipher_name: str, rounds: int = 4) -> Dict[str, Any]:
    """
    执行exp2攻击成功率验证训练
    
    详细描述：使用最优超参数进行模型训练和攻击成功率验证
    
    Args:
        cipher_name (str): 密码算法名称
        rounds (int): 密码轮数，默认为4
        
    Returns:
        Dict[str, Any]: 包含训练结果和攻击成功率的字典
        
    Example:
        >>> train_model("present", 4)
    """
    logger.info(f"开始exp2攻击验证训练 - {cipher_name}, {rounds}轮")
    
    # 加载最优超参数（来自exp1的优化结果）
    best_params = config.get_best_hyperparameters(cipher_name)
    experimental_2 = config.get_experiment_config("exp2")
    
    # 使用TrainModel类进行训练和评估
    trainer = TrainModel(**best_params)

    try:
        # 训练模型（使用exp2的训练数据和测试数据）
        training_results = trainer.train_model(
            data_dir=config.DATA_DIR,
            rounds=rounds,
            cipher_name=cipher_name,
            data_prefix=experimental_2["data_prefix"]
        )

        # 记录攻击成功率验证结果
        success_rate = training_results["training_results"][
            "test_bitwise_success_rate"
        ]

        # 处理success_rate为None的情况
        if success_rate is None:
            success_rate = 0.0
            logger.warning("success_rate为None，设置为0.0")

        equivalent_binary_log = (
            np.log2(success_rate) if success_rate > 0 else float("-inf")
        )
        logger.info(
            f"攻击成功率验证完成 - success_rate: {success_rate:.4f}, "
            f"equivalent_binary_log: {equivalent_binary_log:.4f}, "
            f"cipher: {cipher_name}, rounds: {rounds}, "
            f"best_params: {best_params}"
        )

        return training_results

    except Exception as e:
        logger.error(f"exp2攻击训练失败: {str(e)}")
        raise


def run(cipher_name: str, rounds: int = 4) -> Dict[str, Any]:
    """
    完整运行exp2攻击成功率验证实验

    详细描述：先生成数据，然后进行攻击成功率验证训练

    Args:
        cipher_name (str): 密码算法名称 ('present', 'aes', 'twine')
        rounds (int): 密码轮数，默认为4

    Returns:
        Dict[str, Any]: 包含训练结果和攻击成功率的字典

    Example:
        >>> run("present", 4)
    """
    logger.info(f"开始完整exp2实验 - {cipher_name}, {rounds}轮")
    
    # 生成数据
    generate_data(cipher_name, rounds)
    
    # 训练模型
    training_results = train_model(cipher_name, rounds)
    
    logger.info(f"exp2实验完成 - {cipher_name}, {rounds}轮")
    
    return training_results


if __name__ == "__main__":
    # 运行攻击成功率验证实验
    run("present", 4)
