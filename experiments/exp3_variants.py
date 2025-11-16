import numpy as np
from models.train_model import TrainModel
from ciphers.present import SmallPRESENT4
from utils.data_generator import generate_dataset
from utils.config import config  # [MOD] 配置改为函数式接口
from typing import Dict, Any
from utils.logger import getGlobalLogger

# 创建实验日志器
logger = getGlobalLogger()


class SmallPRESENT4Modified(SmallPRESENT4):
    """组件顺序修改的变体（S盒与置换层交换）"""
    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        state = plaintext.copy()
        round_keys = self._key_schedule(key)
        for r in range(self.rounds - 1):
            state ^= round_keys[r]
            # 先置换层，再S盒（与原始顺序相反）
            state = self._p_layer(state)
            for i in range(4):
                nibble = (state[i*4] << 3) | (state[i*4+1] << 2) | (state[i*4+2] << 1) | state[i*4+3]
                s_nibble = self.S_BOX[nibble]
                # [MOD] 规范续行缩进，避免视觉缩进错误（flake8 E128）
                state[i*4:i*4+4] = [
                    (s_nibble >> 3) & 1, (s_nibble >> 2) & 1,
                    (s_nibble >> 1) & 1, s_nibble & 1,
                ]
        # 最后一轮
        state ^= round_keys[-1]
        state = self._p_layer(state)
        for i in range(4):
            nibble = (state[i*4] << 3) | (state[i*4+1] << 2) | (state[i*4+2] << 1) | state[i*4+3]
            s_nibble = self.S_BOX[nibble]
            # [MOD] 规范续行缩进，避免视觉缩进错误（flake8 E128）
            state[i*4:i*4+4] = [
                (s_nibble >> 3) & 1, (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1, s_nibble & 1,
            ]
        return state


def generate_data(cipher_name: str = "present", rounds: int = 4):
    """
    生成exp3实验所需的训练和测试数据
    
    详细描述：为变体密码攻击实验生成训练和测试数据集
    
    Args:
        cipher_name (str): 密码算法名称，默认为"present"
        rounds (int): 密码轮数，默认为4
        
    Returns:
        None
        
    Example:
        >>> generate_data("present", 4)
    """
    # 初始化组件修改变体密码
    modified_cipher = SmallPRESENT4Modified(rounds=rounds)
    variant_cipher_name = f"component_modification_{cipher_name}"
    
    logger.info(f"开始生成exp3训练数据 - {variant_cipher_name}, {rounds}轮")
    # 生成训练数据集
    # [MOD] 适配新接口：从 exp3 配置读取 keys/train_samples
    exp_cfg = config.getExperimentConfig("exp3")
    generate_dataset(
        cipher=modified_cipher,
        num_keys=exp_cfg.get("keys", 20),
        total_data=exp_cfg.get("train_samples", 16380),
        save_dir=f"{config.getDataDirectory()}/exp3_train_{variant_cipher_name}",
        target_index=exp_cfg.get("target_index", 0)
    )
    
    logger.info(f"开始生成exp3测试数据 - {variant_cipher_name}, {rounds}轮")
    # 生成测试数据集
    generate_dataset(
        cipher=modified_cipher,
        num_keys=exp_cfg.get("keys", 100),
        total_data=exp_cfg.get("test_samples", 32800),
        save_dir=f"{config.getDataDirectory()}/exp3_test_{variant_cipher_name}",
        target_index=exp_cfg.get("target_index", 0)
    )
    
    logger.info(f"exp3数据生成完成 - {variant_cipher_name}, {rounds}轮")


def train_model(cipher_name: str = "present", rounds: int = 4) -> Dict[str, Any]:
    """
    执行exp3变体密码攻击训练
    
    详细描述：使用针对变体优化的超参数进行模型训练和攻击验证
    
    Args:
        cipher_name (str): 密码算法名称，默认为"present"
        rounds (int): 密码轮数，默认为4
        
    Returns:
        Dict[str, Any]: 包含训练结果和攻击成功率的字典
        
    Example:
        >>> train_model("present", 4)
    """
    variant_cipher_name = f"component_modification_{cipher_name}"
    logger.info(f"开始exp3变体攻击训练 - {variant_cipher_name}, {rounds}轮")
    
    # 使用针对变体优化的超参数（可能需要重新优化）
    # 这里先使用标准PRESENT的参数作为基准
    variant_hparams = {
        "hidden_dim": 300,
        "num_layers": 4,
        "lr": 0.01,
        "optimizer": "Adam",
        "batch_size": 250
    }
    
    # 使用TrainModel类进行训练和评估
    trainer = TrainModel(
        hidden_dim=variant_hparams["hidden_dim"],
        num_layers=variant_hparams["num_layers"],
        lr=variant_hparams["lr"],
        optimizer=variant_hparams["optimizer"],
        batch_size=variant_hparams["batch_size"]
    )
    
    try:
        logger.info(f"开始组件修改变体攻击实验 - {variant_cipher_name}, {rounds}轮")
        
        # 训练模型（使用TrainModel类的标准接口）
        training_results = trainer.train_model(
            data_dir=config.getDataDirectory(),
            cipher_name=variant_cipher_name,
            data_prefix="exp3"
        )
        
        # 记录变体攻击实验结果
        success_rate = training_results["training_results"]["test_bitwise_success_rate"]
        
        # 处理success_rate为None的情况
        if success_rate is None:
            success_rate = 0.0
            logger.warning("success_rate为None，设置为0.0")
            
        equivalent_binary_log = (
            np.log2(success_rate) if success_rate > 0 else float('-inf')
        )
        
        logger.info(
            f"变体密码攻击实验完成 - success_rate: {success_rate:.4f}, "
            f"equivalent_binary_log: {equivalent_binary_log:.4f}, "
            f"cipher_variant: {variant_cipher_name}, rounds: {rounds}"
        )
        
        return training_results
        
    except Exception as e:
        logger.error(f"exp3变体攻击训练失败: {str(e)}")
        raise


def run(cipher_name: str = "present", rounds: int = 4) -> Dict[str, Any]:
    """
    完整运行exp3变体密码攻击实验

    详细描述：先生成数据，然后进行变体密码攻击训练

    Args:
        cipher_name (str): 密码算法名称，默认为"present"
        rounds (int): 密码轮数，默认为4

    Returns:
        Dict[str, Any]: 包含训练结果和攻击成功率的字典

    Example:
        >>> run("present", 4)
    """
    logger.info(f"开始完整exp3实验 - {cipher_name}, {rounds}轮")
    
    # 生成数据
    generate_data(cipher_name, rounds)
    
    # 训练模型
    training_results = train_model(cipher_name, rounds)
    
    logger.info(f"exp3实验完成 - {cipher_name}, {rounds}轮")
    
    return training_results


if __name__ == "__main__":
    # 运行变体密码攻击实验
    run("component_modification", 4)