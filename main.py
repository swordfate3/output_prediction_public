import argparse
import os

from experiments.exp1_simplify import (
    run as run_exp1,
    generate_data as generate_exp1_data,
    run_train as train_exp1_model,
)
# [DEL] 删除旧版 exp2_attack 引入，改用简化版接口（保持 run/run_train 语义）
# from experiments.exp2_attack import (
#     run as run_exp2,
#     generate_data as generate_exp2_data,
#     train_model as train_exp2_model
# )
# [ADD] 引入 exp2_simplify：保持 run 端到端与 run_train 单独训练的语义
from experiments.exp2_simplify import (
    run as run_exp2,
    generate_data as generate_exp2_data,
    run_train as train_exp2_model,
)
from experiments.exp3_variants import (
    run as run_exp3,
    generate_data as generate_exp3_data,
    train_model as train_exp3_model
)
import logging
from utils.config import config  # [ADD] 引入配置对象（已改为纯函数接口）

# 配置日志系统
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

logger = logging.getLogger(__name__)


def prepare_directories():
    """创建必要的目录结构。

    详细描述：确保实验运行需要的结果目录存在，包括 `results/exp1`、
    `results/exp2` 与 `results/exp3`。若目录不存在则创建，存在则忽略。

    Args:
        None

    Returns:
        None: 无返回值，功能性创建目录。

    Raises:
        Exception: 当目录创建过程中遇到权限或文件系统错误时可能抛出异常。

    Example:
        >>> prepare_directories()
        >>> # 之后可以安全地写入结果文件到对应目录
    """
    directories = [
        # "data/train", "data/test",
        "results/exp1", "results/exp2", "results/exp3",
        # "models/saved"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("目录结构初始化完成")


def main():
    """命令行入口，分发到各实验的生成与训练逻辑。

    详细描述：解析命令行参数后，根据 `experiment` 与 `mode` 的组合，
    调用相应实验的 `generate_data` 与训练/运行函数。对于 exp1，使用
    简化版 `train_and_test` 统一执行超参数优化与训练，避免重复训练。

    Args:
        None

    Returns:
        None: 无返回值，通过日志与标准输出展示进度与结果。

    Raises:
        SystemExit: 参数解析失败时可能由 argparse 抛出。

    Example:
        >>> # 生成并训练 exp1（present，4轮）
        >>> # 在项目根目录执行：
        >>> # python main.py --experiment exp1 --cipher present \
        >>> # --rounds 4 --mode all
    """
    parser = argparse.ArgumentParser(
        description="论文《Output Prediction Attacks on Block Ciphers "
        "using Deep Learning》实验复现"
    )
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["exp1", "exp2", "exp3"],
                        help="实验编号: exp1(超参数优化), exp2(攻击验证), exp3(变体攻击)")
    parser.add_argument("--cipher", type=str, default="present",
                        help="密码类型: present, aes, twine (exp3仅支持present变体)")
    parser.add_argument("--rounds", type=int, default=4,
                        help="密码轮数 (默认4轮)")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "data", "train"],
        help=(
            "运行模式: all(数据生成+训练), data(仅数据生成), train(仅训练)"
        ),
    )
    
    args = parser.parse_args()

    # 初始化目录
    # prepare_directories()
    
    # 根据模式运行指定实验
    if args.experiment == "exp1":
        # [MOD] 统一为 exp2 风格：all 使用端到端 run，其他模式分别处理
        if args.mode == "all":
            logger.info(
                f"完整运行exp1实验 - cipher: {args.cipher}, "
                f"rounds: {args.rounds}"
            )
            run_exp1(cipher_name=args.cipher, rounds=args.rounds)
        else:
            if args.mode == "data":
                logger.info(
                    f"开始生成exp1数据 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}",
                    f"block_size: {config.getBlockSize()}",
                    f"keys_size: {config.getKeySize()}"
                )
                generate_exp1_data(
                    cipher_name=args.cipher,
                    rounds=args.rounds,
                    block_size=config.getBlockSize(),
                    keys_size=config.getKeySize(),
                )
            elif args.mode == "train":
                logger.info(
                    f"开始exp1超参数优化训练 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}"
                )
                train_exp1_model(
                    cipher_name=args.cipher,
                    rounds=args.rounds,
                    enable_plotting=True,
                )
        
    elif args.experiment == "exp2":
        # [MOD] 简化运行逻辑：all 模式直接使用 run() 端到端，避免重复生成与训练
        if args.mode == "all":
            logger.info(
                f"完整运行exp2实验 - cipher: {args.cipher}, "
                f"rounds: {args.rounds}"
            )
            run_exp2(cipher_name=args.cipher, rounds=args.rounds)
        else:
            if args.mode == "data":
                logger.info(
                    f"开始生成exp2数据 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}"
                )
                # [MOD] 适配新接口：通过函数读取默认尺寸
                generate_exp2_data(
                    cipher_name=args.cipher,
                    rounds=args.rounds,
                    block_size=config.getBlockSize(),
                    keys_size=config.getKeySize(),
                )
            elif args.mode == "train":
                logger.info(
                    f"开始exp2攻击验证训练 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}"
                )
                train_exp2_model(cipher_name=args.cipher, rounds=args.rounds)
        
    elif args.experiment == "exp3":
        # [MOD] 统一为 exp2 风格：all 使用端到端 run，其他模式分别处理
        if args.mode == "all":
            logger.info(
                f"完整运行exp3实验 - cipher: {args.cipher}, "
                f"rounds: {args.rounds}"
            )
            run_exp3(cipher_name=args.cipher, rounds=args.rounds)
        else:
            if args.mode == "data":
                logger.info(
                    f"开始生成exp3数据 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}"
                )
                generate_exp3_data(cipher_name=args.cipher, rounds=args.rounds)
            elif args.mode == "train":
                logger.info(
                    f"开始exp3变体攻击训练 - cipher: {args.cipher}, "
                    f"rounds: {args.rounds}"
                )
                train_exp3_model(cipher_name=args.cipher, rounds=args.rounds)


if __name__ == "__main__":
    main()
