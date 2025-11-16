# -*- coding: utf-8 -*-
"""
exp2_simplify: 使用 TrainModelSimplify 进行攻击成功率风格的训练

- 生成 exp2 数据集（训练/测试）
- 使用 TrainModelSimplify 进行训练与验证
- 评估攻击成功率（bitwise SR 与 Log2 SR）
- 直接使用 config 的 BEST_HYPERPARAMETERS，不需要手动设置
"""
# import time
import os
import argparse
from typing import Dict, Any
import numpy as np

from utils.logger import getGlobalLogger
from utils.config import config  # [MOD] config 接口改为函数式访问
from utils.data_generator import generate_dataset
from models.train_model_simplify import TrainModelSimplify

logger = getGlobalLogger()


def generate_data(cipher_name: str, rounds: int = 4, block_size: int = 128, keys_size: int = 128) -> None:
    """生成 exp2 实验所需的训练和测试数据"""
    # [MOD] 适配新接口：统一从 JSON 获取配置与实例
    cipher = config.getCipherInstance(cipher_name, rounds, block_size, keys_size)
    exp_cfg = config.getExperimentConfig("exp2")

    logger.info(f"开始生成exp2训练数据 - {cipher_name}, {rounds}轮")
    generate_dataset(
        cipher=cipher,
        num_keys=exp_cfg["keys"],
        total_data=exp_cfg["train_samples"],
        save_dir=os.path.join(
            config.getDataDirectory(),
            f"{exp_cfg['data_prefix']}_train_{cipher_name}_round{rounds}",
        ),
        target_index=exp_cfg["target_index"],
        shuffle=True,
    )

    # logger.info(f"开始生成exp2测试数据 - {cipher_name}, {rounds}轮")
    # generate_dataset(
    #     cipher=cipher,
    #     num_keys=exp_cfg["keys"],
    #     total_data=exp_cfg["test_samples"],
    #     save_dir=os.path.join(
    #         config.DATA_DIR,
    #         f"{exp_cfg['data_prefix']}_test_{cipher_name}_round{rounds}"
    #     ),
    #     target_index=exp_cfg["target_index"],
    #     shuffle=True,
    # )

    logger.info(f"exp2数据生成完成 - {cipher_name}, {rounds}轮")


def train_and_report(
    cipher_name: str,
    rounds: int = 4,
    *,
    enable_plotting: bool = True,
) -> Dict[str, Any]:
    """
    使用 TrainModelSimplify 进行训练并报告攻击成功率。

    直接读取 config 中该密码算法的 BEST_HYPERPARAMETERS，
    不再从 CLI 手动设置超参数。
    """
    exp_cfg = config.getExperimentConfig("exp2")
    data_prefix = exp_cfg["data_prefix"]

    # 使用配置中的最佳超参数
    best_hp = config.getBestHyperparameters(cipher_name)
    logger.info(
        (
            f"开始exp2_simplify训练 - cipher: {cipher_name}, rounds: {rounds}, "
            f"使用最佳超参数: {best_hp}, plotting: {enable_plotting}",
            f"使用模型: {config.getModelType()}",
        )
    )

    trainer = TrainModelSimplify(
        hidden_dim=best_hp["hidden_dim"],
        num_layers=best_hp["num_layers"],
        lr=best_hp["lr"],
        batch_size=best_hp["batch_size"],
        epochs=config.getEpochs(),
        optimizer=best_hp["optimizer"],
        criterion=best_hp["criterion"],
        enable_plotting=enable_plotting,
        model_name=config.getModelType(),
        # [ADD] 顶层传递通用 dropout；不同模型的构造将优先使用该值
        dropout=best_hp.get("dropout", 0.1),
        # plot_sr_log_scale=True,
        plot_sr_mixed_scale=True,
        # [ADD] 传递模型特定可选参数；不同模型使用不同键
        model_params={
            # iTransformer 专用：注意力头数
            "num_heads": best_hp.get("num_heads", 4),
            # Mamba 专用：状态空间/卷积/扩展
            "d_state": best_hp.get("d_state", 16),
            "d_conv": best_hp.get("d_conv", 4),
            "expand": best_hp.get("expand", 2),
        },
    )

    train_loader, val_loader, test_loader = trainer.load_data(
        data_dir=config.getDataDirectory(),
        cipher_name=cipher_name,
        rounds=rounds,
        data_prefix=data_prefix,
        val_split=0.5,
        test_split=0.5,
    )

    train_results = trainer.train(train_loader, val_loader)
    test_results = trainer.test(test_loader)

    # 攻击成功率：使用样本级完全匹配成功率（若缺失则回退为比特成功率）
    success_rate = test_results.get("test_sample_sr", None)
    if success_rate is None:
        success_rate = test_results.get("test_bitwise_sr", None)
    if success_rate is None:
        success_rate = 0.0
        logger.warning("success_rate为None，设置为0.0")
    equivalent_binary_log = np.log2(success_rate) if success_rate > 0 else float("-inf")

    logger.info(
        (
            f"攻击成功率验证完成 - success_rate: {success_rate:.4f}, "
            f"equivalent_binary_log: {equivalent_binary_log:.4f}, "
            f"cipher: {cipher_name}, rounds: {rounds}"
        )
    )

    # # 保存模型到 results/exp2 目录
    # save_dir = os.path.join(config.RESULTS_DIR, "exp2")
    # os.makedirs(save_dir, exist_ok=True)
    # ts = time.strftime("%Y%m%d_%H%M%S")
    # model_path = os.path.join(
    #     save_dir,
    #     f"simplify_{cipher_name}_round{rounds}_{ts}.pth",
    # )
    # trainer.save_model(model_path)

    return {
        "train": train_results,
        "test": test_results,
        # "model_path": model_path,
        "success_rate": success_rate,
        "equivalent_binary_log": equivalent_binary_log,
    }


def run(
    cipher_name: str,
    rounds: int = 4,
    *,
    generate: bool = True,
    enable_plotting: bool = True,
) -> Dict[str, Any]:
    """完整运行 exp2_simplify：保持兼容但建议使用子命令"""
    if generate:
        generate_data(cipher_name, rounds)
    return train_and_report(
        cipher_name=cipher_name,
        rounds=rounds,
        enable_plotting=enable_plotting,
    )


def run_generate(cipher_name: str, rounds: int = 4) -> None:
    """仅生成数据（采集流程）"""
    generate_data(cipher_name, rounds)


def run_train(
    cipher_name: str,
    rounds: int = 4,
    *,
    enable_plotting: bool = True,
) -> Dict[str, Any]:
    """仅训练并报告攻击成功率（训练流程）"""
    return train_and_report(
        cipher_name=cipher_name,
        rounds=rounds,
        enable_plotting=enable_plotting,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="exp2_simplify using TrainModelSimplify")

    subparsers = p.add_subparsers(dest="command", required=True)

    # 子命令：generate（仅采集数据）
    gen = subparsers.add_parser("generate", help="生成exp2训练与测试数据")
    gen.add_argument(
        "--cipher",
        type=str,
        default="present",
        # choices=["present", "aes", "twine", "aes128"],
        help="密码算法名称",
    )
    gen.add_argument("--rounds", type=int, default=4, help="密码轮数")

    # 子命令：train（仅训练与评估）
    tr = subparsers.add_parser("train", help="训练并报告攻击成功率")
    tr.add_argument(
        "--cipher",
        type=str,
        default="present",
        # choices=["present", "aes", "twine", "aes128"],
        help="密码算法名称",
    )
    tr.add_argument("--rounds", type=int, default=4, help="密码轮数")
    tr.add_argument(
        "--disable-plotting",
        action="store_true",
        help="禁用绘图功能",
    )

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "generate":
        run_generate(cipher_name=args.cipher, rounds=args.rounds)
    elif args.command == "train":
        results = run_train(
            cipher_name=args.cipher,
            rounds=args.rounds,
            enable_plotting=(not args.disable_plotting),
        )

