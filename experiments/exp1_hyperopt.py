import optuna
from models.train_model import TrainModel
from utils.data_generator import generate_dataset
from utils.config import config  # [MOD] 适配函数式配置接口
from utils.logger import getGlobalLogger
import os
# 创建实验日志器
logger = getGlobalLogger()


def create_objective(cipher_name: str, data_dir: str, rounds: int):
    """
    创建带有自定义参数的objective函数闭包

    Args:
        cipher_name (str): 密码算法名称，用于构建数据路径
        data_dir (str): 数据根目录
        rounds (int): 密码轮数

    Returns:
        function: 配置好的objective函数
    """
    def objective(trial: optuna.Trial):
        """Optuna超参数优化目标函数"""
        # 使用配置的超参数搜索空间
        # [MOD] 使用新接口从 JSON 获取搜索空间
        search_space = config.getHyperparameterSearchSpace()
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", search_space["hidden_dim"]
        )
        num_layers = trial.suggest_int(
            "num_layers",
            search_space["num_layers"]["min"],
            search_space["num_layers"]["max"]
        )
        lr = trial.suggest_categorical("lr", search_space["lr"])
        optimizer_name = trial.suggest_categorical(
            "optimizer", search_space["optimizer"]
        )
        batch_size = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]
        )
        # 使用TrainModel类进行训练
        trainer = TrainModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            optimizer=optimizer_name,
            batch_size=batch_size
        )
        # 训练模型（让TrainModel类处理数据加载和预处理）
        # [MOD] 接口名修正：改为函数式接口 getExperimentConfig()
        exp_cfg = config.getExperimentConfig("exp1")
        training_results = trainer.train_model(
            data_dir=data_dir,
            rounds=rounds,
            cipher_name=cipher_name,
            data_prefix=exp_cfg["data_prefix"]
        )
        # 评估成功率（论文公式：正确预测的密文比例）
        success_rate = training_results["training_results"][
            "test_bitwise_success_rate"
        ]
        return success_rate
    return objective


def generate_data(cipher_name: str, rounds: int):
    """
    生成exp1实验所需的训练和测试数据

    详细描述：为超参数优化实验生成训练和测试数据集

    Args:
        cipher_name (str): 密码算法名称
        rounds (int): 密码轮数

    Returns:
        None

    Example:
        >>> generate_data("present", 4)
    """
    # 使用统一配置获取密码实例与实验配置
    # [MOD] 适配新接口：创建密码实例与读取实验配置
    cipher = config.getCipherInstance(cipher_name, rounds)
    exp_cfg = config.getExperimentConfig("exp1")
    logger.info(f"开始生成exp1训练数据 - {cipher_name}, {rounds}轮")
    generate_dataset(
        cipher=cipher,
        num_keys=exp_cfg["keys"],
        total_data=exp_cfg["train_samples"],
        save_dir=os.path.join(
            config.getDataDirectory(),
            f"{exp_cfg['data_prefix']}_train_{cipher_name}_round{rounds}"
        ),
        target_index=exp_cfg["target_index"],
        shuffle=True
    )
    logger.info(f"开始生成exp1测试数据 - {cipher_name}, {rounds}轮")
    generate_dataset(
        cipher=cipher,
        num_keys=exp_cfg["keys"],
        total_data=exp_cfg["test_samples"],
        save_dir=os.path.join(
            config.getDataDirectory(),
            f"{exp_cfg['data_prefix']}_test_{cipher_name}_round{rounds}"
        ),
        target_index=exp_cfg["target_index"],
        shuffle=True
    )
    logger.info(f"exp1数据生成完成 - {cipher_name}, {rounds}轮")


def train_model(cipher_name: str, rounds: int):
    """
    执行exp1超参数优化训练

    详细描述：使用Optuna进行超参数优化，寻找最佳的模型参数组合

    Args:
        cipher_name (str): 密码算法名称
        rounds (int): 密码轮数

    Returns:
        None

    Example:
        >>> train_model("present", 4)
    """
    logger.info(f"开始exp1超参数优化 - {cipher_name}, {rounds}轮")
    study = optuna.create_study(direction="maximize")
    objective = create_objective(
        cipher_name=cipher_name,
        data_dir=config.getDataDirectory(),
        rounds=rounds
    )
    exp_cfg = config.getExperimentConfig("exp1")
    study.optimize(objective, n_trials=exp_cfg.get("trials", 5))
    logger.info(
        "超参数优化完成",
        extra={
            "best_params": study.best_params,
            "best_value": study.best_value,
            "cipher": cipher_name,
            "rounds": rounds
        }
    )


def run(cipher_name: str, rounds: int):
    """
    完整运行exp1超参数优化实验

    详细描述：先生成数据，然后进行超参数优化训练

    Args:
        cipher_name (str): 密码算法名称
        rounds (int): 密码轮数

    Returns:
        None

    Example:
        >>> run("present", 4)
    """
    logger.info(f"开始完整exp1实验 - {cipher_name}, {rounds}轮")
    # 生成数据
    generate_data(cipher_name, rounds)
    # 训练模型
    train_model(cipher_name, rounds)
    logger.info(f"exp1实验完成 - {cipher_name}, {rounds}轮")


if __name__ == "__main__":
    run("present", 4)