# -*- coding: utf-8 -*-
"""
exp1_simplify: 使用 TrainModelSimplify 进行超参数搜索优化训练与测试

- 生成 exp1 数据集（训练/测试）
- 默认使用 Optuna 进行超参数搜索优化
- 使用最佳参数进行最终训练与验证
- 执行测试并保存模型
- 支持 CLI 参数与可选绘图
"""
import os

# import time
import argparse
from typing import Dict, Any, Callable
import optuna

from utils.logger import getGlobalLogger
from utils.config import config  # [MOD] config 接口已改为函数式访问
from utils.data_generator import generate_dataset
from models.train_model_simplify import TrainModelSimplify

# 为 objective 函数定义类型别名，便于简化函数签名
ObjectiveFn = Callable[[optuna.Trial], float]

logger = getGlobalLogger()


def generate_data(cipher_name: str, rounds: int, block_size: int, keys_size: int) -> None:
    """生成 exp1 实验所需的训练和测试数据"""
    # [MOD] 适配新接口：使用小驼峰式函数
    cipher = config.getCipherInstance(cipher_name, rounds, block_size, keys_size)
    exp_cfg = config.getExperimentConfig("exp1")

    logger.info(f"开始生成exp1训练数据 - {cipher_name}, {rounds}轮")
    generate_dataset(
        cipher=cipher,
        num_keys=exp_cfg["keys"],
        total_data=exp_cfg["train_samples"],
        save_dir=os.path.join(
            config.getDataDirectory(),  # [MOD] 使用新接口获取数据目录
            f"{exp_cfg['data_prefix']}_train_{cipher_name}_round{rounds}",
        ),
        target_index=exp_cfg["target_index"],
        zero_range=exp_cfg["zero_bounds"],
        shuffle=True,
    )

    # logger.info(f"开始生成exp1测试数据 - {cipher_name}, {rounds}轮")
    # generate_dataset(
    #     cipher=cipher,
    #     num_keys=exp_cfg["keys"],
    #     total_data=exp_cfg["test_samples"],
    #     save_dir=os.path.join(
    #         config.getDataDirectory(),  # [MOD] 使用新接口获取数据目录
    #         f"{exp_cfg['data_prefix']}_test_{cipher_name}_round{rounds}"
    #     ),
    #     target_index=exp_cfg["target_index"],
    #     shuffle=True,
    # )

    logger.info(f"exp1数据生成完成 - {cipher_name}, {rounds}轮")


def create_objective(
    cipher_name: str, rounds: int, enable_plotting: bool = False
) -> ObjectiveFn:
    """
    创建带有自定义参数的 objective 函数闭包，适配 TrainModelSimplify。

    详细描述：该函数返回的闭包会根据 Optuna 提供的 Trial，从统一配置中采样
    超参数，构造并训练简化版模型，随后返回验证集上的最终准确率作为优化目标。
    为提高超参搜索效率，objective 内部默认禁用绘图。

    Args:
        cipher_name (str): 密码算法名称，用于构建数据路径。
        rounds (int): 密码轮数。
        enable_plotting (bool): 是否启用绘图功能（[MOD] objective 内部会强制关闭）。

    Returns:
        function: 配置好的 objective 函数闭包。

    Raises:
        Exception: 当数据加载失败或训练过程出错时可能抛出异常（内部已基本捕获并降分处理）。

    Example:
        >>> objective = create_objective("present", 4, enable_plotting=True)
        >>> # 在 Optuna study 中使用
        >>> # study.optimize(objective, n_trials=10)
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna 超参数优化目标函数。

        详细描述：根据统一的搜索空间采样一组超参数，加载数据，进行训练，并
        以验证集最终准确率作为优化目标返回。为了加速搜索与避免生成大量图片，
        [MOD] 此处强制禁用绘图功能，与最终训练流程解耦。

        Args:
            trial (optuna.Trial): Optuna 提供的试验实例，用于采样超参数。

        Returns:
            float: 验证集最终准确率（final_val_acc），用于最大化目标。

        Raises:
            Exception: 不直接抛出，内部捕获后返回 0.0 以标记失败试验。

        Example:
            >>> # 在 Optuna 调用中由框架传入 trial
            >>> score = objective(trial)
            >>> assert isinstance(score, float)
        """
        # 使用配置的超参数搜索空间
        # [MOD] 适配新接口：统一从 JSON 获取搜索空间
        search_space = config.getHyperparameterSearchSpace()
        hidden_dim = trial.suggest_categorical("hidden_dim", search_space["hidden_dim"])
        num_layers = trial.suggest_int(
            "num_layers",
            search_space["num_layers"]["min"],
            search_space["num_layers"]["max"],
        )
        lr = trial.suggest_categorical("lr", search_space["lr"])
        optimizer = trial.suggest_categorical("optimizer", search_space["optimizer"])
        criterion = trial.suggest_categorical("criterion", search_space["criterion"])
        threshold = trial.suggest_categorical("threshold", search_space["threshold"])
        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])
        dropout = trial.suggest_categorical("dropout", search_space["dropout"])
        trainer = None
        # [MOD] 适配新接口：模型类型通过函数读取
        if config.getModelType() == "mamba":
            d_state = trial.suggest_categorical("d_state", search_space["d_state"])
            d_conv = trial.suggest_categorical("d_conv", search_space["d_conv"])
            expand = trial.suggest_categorical("expand", search_space["expand"])
            trainer = TrainModelSimplify(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                lr=lr,
                optimizer=optimizer,
                criterion=criterion,
                threshold=threshold,
                batch_size=batch_size,
                epochs=config.getEpochs(),  # [MOD] 从 JSON 读取训练轮数
                model_name=config.getModelType(),
                enable_plotting=False,  # [MOD] 关闭绘图，加速超参搜索
                # [ADD] 顶层传递通用 dropout（若搜索空间包含则采样使用）
                dropout=dropout,
                # [ADD] 传递模型特定可选参数；不同模型使用不同键
                model_params={
                    # Mamba 专用：状态空间/卷积/扩展
                    "d_state": d_state,
                    "d_conv": d_conv,
                    "expand": expand,
                },
            )
        elif config.getModelType() == "itransformer":
            num_heads = trial.suggest_categorical(
                "num_heads", search_space["num_heads"]
            )
            trainer = TrainModelSimplify(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                lr=lr,
                optimizer=optimizer,
                criterion=criterion,
                threshold=threshold,
                batch_size=batch_size,
                epochs=config.getEpochs(),  # [MOD] 从 JSON 读取训练轮数
                model_name=config.getModelType(),
                enable_plotting=False,  # [MOD] 关闭绘图，加速超参搜索
                # [ADD] 顶层传递通用 dropout（若搜索空间包含则采样使用）
                dropout=dropout,
                # [ADD] 传递模型特定可选参数；不同模型使用不同键
                model_params={
                    # iTransformer 专用：头数
                    "num_heads": num_heads,
                },
            )
        else:
            trainer = TrainModelSimplify(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                lr=lr,
                optimizer=optimizer,
                criterion=criterion,
                threshold=threshold,
                batch_size=batch_size,
                epochs=config.getEpochs(),  # [MOD] 从 JSON 读取训练轮数
                model_name=config.getModelType(),
                enable_plotting=False,  # [MOD] 关闭绘图，加速超参搜索
                # [ADD] 顶层传递通用 dropout（若搜索空间包含则采样使用）
                dropout=dropout,
            )

        # # [MOD] objective 中统一禁用绘图以提升搜索效率；保持与最终训练流程参数一致
        # model_name = config.MODEL_TYPE
        # # 使用 TrainModelSimplify 类进行训练
        # trainer = TrainModelSimplify(
        #     hidden_dim=hidden_dim,
        #     num_layers=num_layers,
        #     lr=lr,
        #     optimizer=optimizer,
        #     criterion=criterion,
        #     threshold=threshold,
        #     batch_size=batch_size,
        #     epochs=config.EPOCHS,  # 固定 epochs 为配置值
        #     model_name=model_name,
        #     enable_plotting=False,  # [MOD] 关闭绘图，加速超参搜索
        #     # [ADD] 顶层传递通用 dropout（若搜索空间包含则采样使用）
        #     dropout=dropout,
        # )  # type: ignore

        # 加载数据
        # [MOD] 接口名修正：改为函数式接口 getExperimentConfig()
        exp_cfg = config.getExperimentConfig("exp1")
        data_prefix = exp_cfg["data_prefix"]

        try:
            train_loader, val_loader, test_loader = trainer.load_data(
                data_dir=config.getDataDirectory(),  # [MOD] 使用新接口获取数据目录
                cipher_name=cipher_name,
                rounds=rounds,
                data_prefix=data_prefix,
            )
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            # 返回一个很低的分数表示失败
            return 0.0

        # 训练模型并直接使用返回的最终指标作为优化目标
        # [MOD] 与最终训练流程对齐：使用 train() 的结果字典，而非再次调用 validate()
        train_results = trainer.train(train_loader, val_loader)
        success_rate = float(train_results.get("final_sample_sr", 0.0))

        # [DEL] 不再单独调用 validate() 并解包其 6 个返回值，避免重复计算与潜在解包错误
        #       原逻辑删除后，指标改为从 train() 的返回字典读取。

        # [ADD] 统一日志输出：与最终训练流程的 key 命名保持一致，便于比较
        logger.info(
            (
                f"Trial {trial.number}: hidden_dim={hidden_dim}, "
                f"model_name={config.getModelType()}, "
                f"num_layers={num_layers}, lr={lr}, "
                f"batch_size={batch_size}, optimizer={optimizer}, "
                f"criterion={criterion}, threshold={threshold}, "
                f"final_val_acc={train_results.get('final_val_acc', 0.0):.4f}, "
                f"final_bitwise_sr={train_results.get('final_bitwise_sr', 0.0):.4f}, "
                f"final_log2_sr={train_results.get('final_log2_sr', 0.0):.4f}, "
                f"final_sample_sr={train_results.get('final_sample_sr', 0.0):.4f}, "
                f"final_bit_match_pct={train_results.get('final_bit_match_pct', 0.0):.4f}"
            )
        )

        return success_rate

    return objective


def train_and_test(
    cipher_name: str,
    rounds: int,
    *,
    epochs: int = 20,
    enable_plotting: bool = True,
    trials: int = 10,
) -> Dict[str, Any]:
    """使用 TrainModelSimplify 进行超参数搜索优化训练、验证与测试"""
    # [MOD] 接口名修正：改为函数式接口 getExperimentConfig()
    exp_cfg = config.getExperimentConfig("exp1")
    data_prefix = exp_cfg["data_prefix"]

    # 超参数搜索模式（默认模式）
    logger.info(
        f"开始exp1_simplify超参数优化 - cipher: {cipher_name}, "
        f"rounds: {rounds}, trials: {trials}, plotting: {enable_plotting}, model: {config.getModelType()}"
    )

    # 创建Optuna study
    study = optuna.create_study(direction="maximize")
    objective = create_objective(
        cipher_name=cipher_name, rounds=rounds, enable_plotting=enable_plotting
    )

    # 执行超参数优化
    study.optimize(objective, n_trials=trials)

    # 记录最佳结果
    logger.info(
        "超参数优化完成",
        extra={
            "best_params": study.best_params,
            "best_value": study.best_value,
            "cipher": cipher_name,
            "rounds": rounds,
        },
    )

    # 使用最佳参数进行最终训练
    best_params = study.best_params
    trainer = None
    if config.getModelType() == "mamba":
        trainer = TrainModelSimplify(
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            lr=best_params["lr"],
            optimizer=best_params.get("optimizer", "Adam"),
            criterion=best_params.get("criterion", "BCELoss"),
            threshold=best_params.get("threshold", 0.5),
            batch_size=best_params["batch_size"],
            epochs=config.getEpochs(),  # [MOD] 使用新接口获取 epochs
            enable_plotting=enable_plotting,
            model_name=config.getModelType(),  # [MOD] 使用新接口获取模型类型
            # [ADD] 顶层传递通用 dropout；若无则回退默认值
            dropout=best_params.get("dropout", 0.1),
            # [ADD] 传递模型特定可选参数；可由搜索空间或默认值提供
            model_params={
                "d_state": best_params.get("d_state", 16),
                "d_conv": best_params.get("d_conv", 4),
                "expand": best_params.get("expand", 2),
            },
        )
    elif config.getModelType() == "itransformer":
        trainer = TrainModelSimplify(
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            lr=best_params["lr"],
            optimizer=best_params.get("optimizer", "Adam"),
            criterion=best_params.get("criterion", "BCELoss"),
            threshold=best_params.get("threshold", 0.5),
            batch_size=best_params["batch_size"],
            epochs=config.getEpochs(),  # [MOD] 使用新接口获取 epochs
            enable_plotting=enable_plotting,
            model_name=config.getModelType(),  # [MOD] 使用新接口获取模型类型
            # [ADD] 顶层传递通用 dropout；若无则回退默认值
            dropout=best_params.get("dropout", 0.1),
            # [ADD] 传递模型特定可选参数；可由搜索空间或默认值提供
            model_params={
                "num_heads": best_params.get("num_heads", 4),
            },
        )
    else:
        trainer = TrainModelSimplify(
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            lr=best_params["lr"],
            optimizer=best_params.get("optimizer", "Adam"),
            criterion=best_params.get("criterion", "BCELoss"),
            threshold=best_params.get("threshold", 0.5),
            batch_size=best_params["batch_size"],
            epochs=config.getEpochs(),  # [MOD] 使用新接口获取 epochs
            enable_plotting=enable_plotting,
            model_name=config.getModelType(),  # [MOD] 使用新接口获取模型类型
            # [ADD] 顶层传递通用 dropout；若无则回退默认值
            dropout=best_params.get("dropout", 0.1),
        )

    logger.info(f"使用最佳参数进行最终训练: {best_params}")

    try:
        train_loader, val_loader, test_loader = trainer.load_data(
            data_dir=config.getDataDirectory(),
            cipher_name=cipher_name,
            rounds=rounds,
            data_prefix=data_prefix,
        )
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

    train_results = trainer.train(train_loader, val_loader)
    test_results = trainer.test(test_loader)

    # 保存模型到 results/exp1 目录
    # save_dir = os.path.join(config.RESULTS_DIR, "exp1")
    # os.makedirs(save_dir, exist_ok=True)
    # ts = time.strftime("%Y%m%d_%H%M%S")
    # model_path = os.path.join(
    #     save_dir,
    #     f"simplify_{cipher_name}_round{rounds}_{ts}.pth",
    # )
    # trainer.save_model(model_path)
    # logger.info(f"模型已保存: {model_path}")

    # 汇总输出
    # logger.info("训练完成 - 指标：")
    # logger.info(f"  final_val_acc={train_results['final_val_acc']:.4f}")
    # logger.info(f"  final_bitwise_sr={train_results['final_bitwise_sr']:.4f}")
    # logger.info(f"  final_log2_sr={train_results['final_log2_sr']:.4f}")
    # logger.info(f"  final_sample_sr={train_results['final_sample_sr']:.4f}")

    # logger.info("测试结果：")
    # logger.info(f"  acc={test_results['test_acc']:.4f}")
    # logger.info(f"  bitwise_sr={test_results['test_bitwise_sr']:.4f}")
    # logger.info(f"  log2_sr={test_results['test_log2_sr']:.4f}")
    # logger.info(f"  sample_sr={test_results['test_sample_sr']:.4f}")

    return {"train": train_results, "test": test_results}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="exp1_simplify：超参数搜索优化训练的CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="子命令：generate 或 train", required=True
    )

    # generate 子命令
    gen = subparsers.add_parser("generate", help="生成exp1数据集")
    gen.add_argument(
        "--cipher",
        type=str,
        default="present",
        # choices=["present", "aes", "twine", "aes128"],
        help="密码算法名称",
    )
    gen.add_argument("--rounds", type=int, default=4, help="密码轮数")

    # train 子命令（超参数搜索优化训练）
    train = subparsers.add_parser(
        "train", help="超参数搜索优化训练与测试（不生成数据）"
    )
    train.add_argument(
        "--cipher",
        type=str,
        default="present",
        # choices=["present", "aes", "twine", "aes128"],
        help="密码算法名称",
    )
    train.add_argument("--rounds", type=int, default=4, help="密码轮数")
    train.add_argument("--epochs", type=int, default=20, help="训练轮数")
    train.add_argument(
        "--disable-plotting",
        action="store_true",
        help="禁用绘图功能",
    )
    # train.add_argument(
    #     "--trials",
    #     type=int,
    #     default=10,
    #     help="超参数优化试验次数（默认10次）",
    # )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    exp_cfg = config.get_experiment_config("exp1")
    if args.command == "generate":
        generate_data(args.cipher, args.rounds, exp_cfg["block_size"], exp_cfg["keys_size"])
        logger.info("数据生成完成")
        return

    if args.command == "train":
        train_and_test(
            cipher_name=args.cipher,
            rounds=args.rounds,
            epochs=args.epochs,
            enable_plotting=(not args.disable_plotting),
            trials=exp_cfg["trials"],
        )
        # # 简要打印最终结果
        # tr = results["train"]
        # te = results["test"]
        # print("训练验证指标：")
        # print(f"  final_val_acc={tr['final_val_acc']:.4f}")
        # print(f"  final_bitwise_sr={tr['final_bitwise_sr']:.4f}")
        # print(f"  final_log2_sr={tr['final_log2_sr']:.4f}")
        # print("测试指标：")
        # print(f"  test_acc={te['test_acc']:.4f}")
        # print(f"  test_bitwise_sr={te['test_bitwise_sr']:.4f}")
        # print(f"  test_log2_sr={te['test_log2_sr']:.4f}")
        return


if __name__ == "__main__":
    main()
