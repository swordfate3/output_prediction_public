# 导入必要的库和模块
import torch
import numpy as np
import os
from utils.metrics import bitwise_success_rate, log2_success_rate
from utils.config import config
from callbacks import EarlyStoppingCallback
from callbacks.plotting import PlottingCallback
from callbacks.base_callback import CallbackManager
from models.lstm_model import CipherLSTM
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import getGlobalLogger

# 创建实验日志器
logger = getGlobalLogger()
# 导入自定义组件（可选）
try:
    from custom_components.optimizers import get_custom_optimizer
    from custom_components.losses import get_custom_loss

    CUSTOM_COMPONENTS_AVAILABLE = False
except ImportError:
    CUSTOM_COMPONENTS_AVAILABLE = False


class TrainModel:
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "cpu")
        # 直接存储解包后的参数字典
        self.train_parameters = kwargs
        # 初始化训练结果记录
        self.training_results = self._initialize_training_results()

    def _initialize_training_results(self):
        """
        初始化训练结果字典

        创建用于存储训练过程中各种指标的字典结构

        Returns:
            dict: 初始化的训练结果字典

        Example:
            >>> results = self._initialize_training_results()
        """
        return {
            # 历史记录列表
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "learning_rates": [],
            "gradient_norms": [],
            "bitwise_success_rates": [],
            "log2_success_rates": [],
            # 当前值
            "train_loss": None,
            "train_acc": None,
            "val_loss": None,
            "val_acc": None,
            "learning_rate": None,
            "gradient_norm": None,
            "bitwise_success_rate": None,
            "log2_success_rate": None,
            # 训练状态
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "stopped_early": False,
            "total_epochs": 0,
            "success_rate": None,
            # 测试集评估结果
            "test_loss": None,
            "test_acc": None,
            "test_bitwise_success_rate": None,
            "test_log2_success_rate": None,
        }

    def _prepare_data(self, data_dir, cipher_name, rounds=4,
                      data_prefix="exp1"):
        """
        准备训练数据

        从文件加载数据，处理验证集划分和张量转换等数据预处理步骤

        Args:
            data_dir (str): 数据目录路径
            cipher_name (str): 密码算法名称，用于构建文件路径
            data_prefix (str): 数据路径前缀，默认为"exp1"

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test) 处理后的数据张量

        Raises:
            ValueError: 当数据参数不符合要求时
            FileNotFoundError: 当数据文件不存在时
            RuntimeError: 当数据处理过程中发生错误时

        Example:
            >>> X_train, y_train, X_val, y_val, X_test, y_test = 
                self._prepare_data(
            ...     data_dir="./data", cipher_name="present", 
                    data_prefix="exp1"
            ... )
        """
        try:
            # 保存数据目录和密码名称以供后续使用
            self.data_dir = data_dir
            self.cipher_name = cipher_name
            self.data_prefix = data_prefix

            # 从文件加载数据
            try:
                X_train, y_train, X_test, y_test = self._load_data_from_files(
                    data_dir, cipher_name, rounds, data_prefix
                )
            except FileNotFoundError as e:
                logger.error(f"数据文件不存在: {str(e)}")
                raise

            # 划分验证集
            X_train, X_val, y_train, y_val = self._split_validation_data(
                X_train, y_train
            )

            # 转换为张量格式
            (X_train, y_train, X_val, y_val,
             X_test, y_test) = self._convert_to_tensors(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

            return X_train, y_train, X_val, y_val, X_test, y_test
        except Exception as e:
            logger.error(f"数据准备失败: {str(e)}")
            raise RuntimeError(f"数据准备过程中发生错误: {str(e)}") from e

    def _load_data_from_files(
        self, data_dir, cipher_name, rounds=4, data_prefix="exp1"
    ):
        """
        从文件加载数据

        从指定目录加载训练和测试数据文件

        Args:
            data_dir (str): 数据目录路径
            cipher_name (str): 密码算法名称
            data_prefix (str): 数据路径前缀，默认为"exp1"

        Returns:
            tuple: (X_train, y_train, X_test, y_test) 加载的数据数组

        Raises:
            FileNotFoundError: 当数据文件不存在时
            ValueError: 当数据格式不正确时

        Example:
            >>> X_train, y_train, X_test, y_test = self._load_data_from_files(
            ...     "./data", "present", "exp1"
            ... )
        """
        try:
            train_path = (
                f"{data_dir}/{data_prefix}_train_{cipher_name}_round{rounds}"
            )
            test_path = (
                f"{data_dir}/{data_prefix}_test_{cipher_name}_round{rounds}"
            )

            # 检查文件是否存在

            if not os.path.exists(f"{train_path}/plain_texts.npy"):
                logger.warning(f"训练数据文件不存在: {train_path}/plain_texts.npy")
                raise FileNotFoundError(
                    f"训练数据文件不存在: {train_path}/plain_texts.npy"
                )
            if not os.path.exists(f"{train_path}/cipher_texts.npy"):
                logger.warning(f"训练标签文件不存在: {train_path}/cipher_texts.npy")
                raise FileNotFoundError(
                    f"训练标签文件不存在: {train_path}/cipher_texts.npy"
                )

            # 加载训练数据
            X_train = np.load(f"{train_path}/plain_texts.npy")
            y_train = np.load(f"{train_path}/cipher_texts.npy")

            # 验证数据形状
            if X_train.shape[0] != y_train.shape[0]:
                logger.error(
                    f"训练数据样本数不匹配: X_train={X_train.shape[0]}, "
                    f"y_train={y_train.shape[0]}"
                )
                raise ValueError(
                    f"训练数据样本数不匹配: X_train={X_train.shape[0]}, "
                    f"y_train={y_train.shape[0]}"
                )

            # 加载测试数据
            if not os.path.exists(f"{test_path}/plain_texts.npy"):
                logger.warning(f"测试数据文件不存在: {test_path}/plain_texts.npy")
                X_test, y_test = None, None
            elif not os.path.exists(f"{test_path}/cipher_texts.npy"):
                logger.warning(f"测试标签文件不存在: {test_path}/cipher_texts.npy")
                X_test, y_test = None, None
            else:
                X_test = np.load(f"{test_path}/plain_texts.npy")
                y_test = np.load(f"{test_path}/cipher_texts.npy")

                # 验证测试数据形状
                if X_test.shape[0] != y_test.shape[0]:
                    logger.error(
                        f"测试数据样本数不匹配: X_test={X_test.shape[0]}, "
                        f"y_test={y_test.shape[0]}"
                    )
                    raise ValueError(
                        f"测试数据样本数不匹配: X_test={X_test.shape[0]}, "
                        f"y_test={y_test.shape[0]}"
                    )

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"数据文件加载失败: {str(e)}")
            raise

    def _split_validation_data(self, X_train, y_train, split_ratio=0.8):
        """
        划分训练集和验证集

        将训练数据按指定比例划分为训练集和验证集

        Args:
            X_train: 训练输入数据
            y_train: 训练标签数据
            split_ratio (float): 训练集占比，默认0.8

        Returns:
            tuple: (X_train, X_val, y_train, y_val) 划分后的数据

        Example:
            >>> X_train, X_val, y_train, y_val = self._split_validation_data(
            ...     X_train, y_train, split_ratio=0.8
            ... )
        """
        train_size = int(split_ratio * len(X_train))
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

        return X_train, X_val, y_train, y_val

    def _convert_to_tensors(
        self, X_train, y_train, X_val, y_val, X_test, y_test
    ):
        """
        将数据转换为PyTorch张量格式

        统一处理所有数据的张量转换，确保正确的数据类型和维度

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: 输入数据

        Returns:
            tuple: 转换后的张量数据

        Example:
            >>> tensors = self._convert_to_tensors(
            ...     X_train, y_train, X_val, y_val, X_test, y_test
            ... )
        """

        def convert_to_tensor(data, is_label=False):
            """统一的张量转换函数"""
            if data is None:
                return None

            try:
                if not isinstance(data, torch.Tensor):
                    tensor = (
                        torch.from_numpy(data)
                        if isinstance(data, np.ndarray)
                        else torch.tensor(data)
                    )
                    tensor = tensor.float()
                else:
                    tensor = data.float()

                # 处理标签维度 - 确保标签是二维的 (batch_size, output_dim)
                if is_label:
                    if len(tensor.shape) == 1:
                        tensor = tensor.unsqueeze(-1)
                    elif len(tensor.shape) > 2:
                        logger.warning(
                            f"标签张量维度过高: {tensor.shape}，将重塑为二维"
                        )
                        tensor = tensor.view(tensor.shape[0], -1)

                # 为输入数据处理维度 - LSTM期望 (batch_size, seq_len, input_size)
                if not is_label:
                    if len(tensor.shape) == 1:
                        # 单个样本，添加batch和sequence维度
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    elif len(tensor.shape) == 2:
                        # 批次数据，添加sequence维度
                        tensor = tensor.unsqueeze(1)
                    elif len(tensor.shape) > 3:
                        logger.warning(
                            f"输入张量维度过高: {tensor.shape}，将重塑为三维"
                        )
                        tensor = tensor.view(tensor.shape[0], 1, -1)

                return tensor
            except Exception as e:
                logger.error(f"张量转换失败: {str(e)}")
                raise ValueError(f"无法转换数据为张量: {str(e)}") from e

        return (
            convert_to_tensor(X_train),
            convert_to_tensor(y_train, is_label=True),
            convert_to_tensor(X_val),
            convert_to_tensor(y_val, is_label=True),
            convert_to_tensor(X_test),
            convert_to_tensor(y_test, is_label=True),
        )

    def _create_data_loaders(self, X_train, y_train, X_val, y_val):
        """
        创建训练和验证数据加载器

        将张量数据封装为DataLoader对象，便于批量训练

        Args:
            X_train: 训练输入张量
            y_train: 训练标签张量
            X_val: 验证输入张量
            y_val: 验证标签张量

        Returns:
            tuple: (train_loader, val_loader) 数据加载器对象

        Example:
            >>> train_loader, val_loader = self._create_data_loaders(
            ...     X_train, y_train, X_val, y_val
            ... )
        """
        batch_size = self.train_parameters.get("batch_size", 32)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader

    def _setup_model(self, model_config):
        """
        设置模型、优化器和损失函数

        根据统一配置初始化模型、优化器和损失函数，简化设置流程

        Args:
            model_config (dict): 统一配置参数，包含模型、优化器和损失函数配置
                - 模型配置: input_size, hidden_size, num_layers, output_size, 
                           dropout
                - 优化器配置: optimizer.type, optimizer.learning_rate, momentum
                - 损失函数配置: criterion.type, criterion.reduction, weight等

        Returns:
            tuple: (模型, 优化器, 损失函数)

        Example:
            >>> model, optimizer, criterion = (
            ...     self._setup_model(model_config)
            ... )
        """
        # 初始化模型
        model = self._initialize_model(model_config)

        # 从统一配置中提取优化器配置
        optimizer = self._initialize_optimizer(model, model_config)

        # 从统一配置中提取损失函数配置
        criterion = self._initialize_criterion(model_config)

        return model, optimizer, criterion

    def _initialize_optimizer(self, model, model_config):
        """
        初始化优化器

        根据配置参数创建并初始化优化器，支持自定义优化器

        Args:
            model: 神经网络模型
            model_config (dict): 模型配置参数

        Returns:
            torch.optim.Optimizer: 初始化的优化器

        Example:
            >>> optimizer = self._initialize_optimizer(model, {
            ...     "optimizer": "adamw_custom",
            ...     "learning_rate": 0.001,
            ...     "weight_decay": 0.01
            ... })
        """
        optimizer_type = model_config.get("optimizer", "adam").lower()
        lr = model_config.get("learning_rate", 0.001)

        # 尝试使用自定义优化器
        if CUSTOM_COMPONENTS_AVAILABLE:
            try:
                custom_optimizer = get_custom_optimizer(optimizer_type)
                optimizer_config = model_config.copy()
                optimizer_config["learning_rate"] = lr
                return custom_optimizer.create_optimizer(
                    model.parameters(), optimizer_config
                )
            except ValueError:
                # 如果不是自定义优化器，继续使用内置优化器
                pass

        # 内置优化器
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            momentum = model_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum
            )
        elif optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_type == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        else:
            logger.warning(f"未知优化器类型 {optimizer_type}，使用默认Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return optimizer

    def _initialize_criterion(self, model_config):
        """
        初始化损失函数

        根据配置参数创建并初始化损失函数，支持自定义损失函数

        Args:
            criterion_config (dict): 损失函数配置参数，可选
                - type (str): 损失函数类型，支持内置和自定义类型
                - weight (tensor): 类别权重，可选
                - reduction (str): 损失计算方式，'mean', 'sum', 'none'
                - pos_weight (tensor): BCEWithLogitsLoss正样本权重
                - alpha (float): Focal Loss平衡因子
                - gamma (float): Focal Loss聚焦参数

        Returns:
            torch.nn.Module: 初始化的损失函数

        Example:
            >>> criterion = self._initialize_criterion({
            ...     'type': 'focal_loss',
            ...     'alpha': 1.0,
            ...     'gamma': 2.0
            ... })
        """

        criterion_config = model_config.get("criterion", "bce")
        if isinstance(criterion_config, str):
            criterion_type = criterion_config.lower()
        else:
            criterion_type = criterion_config.get("type", "BCELoss").lower()

        # 尝试使用自定义损失函数
        if CUSTOM_COMPONENTS_AVAILABLE:
            try:
                custom_loss = get_custom_loss(criterion_type)
                loss_config = (
                    (criterion_config
                     if isinstance(criterion_config, dict)
                     else {})
                )
                return custom_loss.create_loss(loss_config)
            except ValueError:
                # 如果不是自定义损失函数，继续使用内置损失函数
                pass

        # 内置损失函数
        if criterion_type == "bceloss":
            criterion = nn.BCELoss()
        elif criterion_type == "mseloss":
            criterion = nn.MSELoss()
        elif criterion_type == "cross_entropyloss":
            criterion = nn.CrossEntropyLoss()
        elif criterion_type == "l1loss":
            criterion = nn.L1Loss()
        elif criterion_type == "smooth_l1loss":
            criterion = nn.SmoothL1Loss()
        else:
            logger.warning(f"未知损失函数类型 {criterion_type}，使用默认BCELoss")
            criterion = nn.BCELoss()

        return criterion

    def _initialize_model(self, model_config):
        """
        初始化神经网络模型

        根据配置参数创建并初始化模型

        Args:
            model_config (dict): 模型配置参数

        Returns:
            torch.nn.Module: 初始化的模型

        Example:
            >>> model = self._initialize_model(model_config)
        """
        input_dim = model_config.get("input_size", 32)  # 映射到 input_dim
        hidden_dim = model_config.get("hidden_size", 64)  # 映射到 hidden_dim
        output_dim = model_config.get("output_size", 32)  # 映射到 output_dim
        num_layers = model_config.get("num_layers", 2)
        dropout = model_config.get("dropout", 0.1)

        model = CipherLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        return model

    def _process_batch(
        self, model, inputs, labels, criterion, optimizer=None, 
        is_training=True
    ):
        """
        处理单个批次的通用方法

        统一处理训练和验证批次，减少代码重复

        Args:
            model: 神经网络模型
            inputs: 输入数据
            labels: 标签数据
            criterion: 损失函数
            optimizer: 优化器（仅训练时使用）
            is_training (bool): 是否为训练模式

        Returns:
            dict: 包含损失、预测结果等的字典

        Example:
            >>> batch_result = self._process_batch(
            ...     model, inputs, labels, criterion, optimizer, True
            ... )
        """
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 检查损失异常
        if torch.isnan(loss) or torch.isinf(loss):
            return {"skip": True, "loss": loss.item()}

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 计算预测结果
        predicted = (outputs > 0.5).float()
        correct = (predicted == labels).sum().item()
        total_samples = labels.numel()

        return {
            "skip": False,
            "loss": loss.item(),
            "correct": correct,
            "total_samples": total_samples,
            "predictions": predicted.cpu().numpy()
            if not is_training
            else None,
            "targets": labels.cpu().numpy() if not is_training else None,
        }

    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """
        执行单个训练轮次

        在训练数据上执行一个完整的前向传播、反向传播和参数更新过程

        Args:
            model: 要训练的神经网络模型
            train_loader: 训练数据加载器
            optimizer: 优化器对象
            criterion: 损失函数

        Returns:
            tuple: (平均训练损失, 训练准确率)

        Example:
            >>> train_loss, train_acc = self._train_epoch(
            ...     model, train_loader, optimizer, criterion
            ... )
        """
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                batch_result = self._process_batch(
                    model, inputs, labels, criterion, optimizer, True
                )

                if batch_result["skip"]:
                    logger.warning(
                        f"批次 {batch_idx} 出现异常损失值: {batch_result['loss']}"
                    )
                    continue

                total_loss += batch_result["loss"]
                total_correct += batch_result["correct"]
                total_samples += batch_result["total_samples"]

            except Exception as e:
                logger.warning(f"批次 {batch_idx} 训练失败: {str(e)}")
                continue

        if total_samples == 0:
            raise RuntimeError("没有成功处理任何训练批次")

        return total_loss / len(train_loader), total_correct / total_samples

    def _validate_epoch(self, model, val_loader, criterion):
        """
        执行单个验证轮次

        在验证数据上评估模型性能，计算各种指标

        Args:
            model: 要验证的神经网络模型
            val_loader: 验证数据加载器
            criterion: 损失函数

        Returns:
            dict: 包含验证指标的字典

        Example:
            >>> val_metrics = self._validate_epoch(model, val_loader, 
                                                   criterion)
        """
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                try:
                    batch_result = self._process_batch(
                        model, inputs, labels, criterion, None, False
                    )

                    if batch_result["skip"]:
                        logger.warning(
                            f"验证批次 {batch_idx} 出现异常损失值: {batch_result['loss']}"
                        )
                        continue

                    total_loss += batch_result["loss"]
                    total_correct += batch_result["correct"]
                    total_samples += batch_result["total_samples"]

                    # 安全地处理预测和目标数据
                    if batch_result["predictions"] is not None:
                        predictions = batch_result["predictions"]
                        if predictions.ndim > 2:
                            predictions = predictions.reshape(
                                -1, predictions.shape[-1]
                            )
                        all_predictions.append(predictions)

                    if batch_result["targets"] is not None:
                        targets = batch_result["targets"]
                        if targets.ndim > 2:
                            targets = targets.reshape(-1, targets.shape[-1])
                        all_targets.append(targets)

                except Exception as e:
                    logger.warning(f"验证批次 {batch_idx} 处理失败: {str(e)}")
                    continue

        if total_samples == 0:
            raise RuntimeError("没有成功处理任何验证批次")

        # 计算成功率指标
        try:
            if all_predictions and all_targets:
                # 正确拼接数组
                predictions_array = np.concatenate(all_predictions, axis=0)
                targets_array = np.concatenate(all_targets, axis=0)

                predictions_tensor = torch.tensor(predictions_array)
                targets_tensor = torch.tensor(targets_array)

                bitwise_sr = bitwise_success_rate(
                    predictions_tensor, targets_tensor
                )
                log2_sr = log2_success_rate(bitwise_sr)
            else:
                logger.warning("没有有效的预测或目标数据用于计算成功率")
                bitwise_sr = log2_sr = 0.0
        except Exception as e:
            logger.warning(f"成功率计算失败: {str(e)}")
            bitwise_sr = log2_sr = 0.0

        return {
            "val_loss": total_loss / len(val_loader),
            "val_acc": total_correct / total_samples,
            "bitwise_success_rate": bitwise_sr,
            "log2_success_rate": log2_sr,
        }

    def _update_training_results(
        self, train_loss, train_acc, val_metrics, optimizer, model=None
    ):
        """
        更新训练结果记录

        将当前轮次的训练和验证指标添加到结果记录中

        Args:
            train_loss: 训练损失
            train_acc: 训练准确率
            val_metrics: 验证指标字典
            optimizer: 优化器对象，用于获取学习率
            model: 神经网络模型，用于计算梯度范数

        Example:
            >>> self._update_training_results(
            ...     train_loss, train_acc, val_metrics, optimizer, model
            ... )
        """
        # 计算额外指标
        current_lr = optimizer.param_groups[0]["lr"]
        gradient_norm = self._calculate_gradient_norm(model)

        # 准备更新数据
        update_data = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "learning_rate": current_lr,
            "gradient_norm": gradient_norm,
            "bitwise_success_rate": val_metrics["bitwise_success_rate"],
            "log2_success_rate": val_metrics["log2_success_rate"],
        }

        # 添加到历史记录
        for key, value in update_data.items():
            history_key = (
                f"{key}s" if key.endswith(("loss", "acc")) else f"{key}s"
            )
            if key == "train_loss":
                history_key = "train_losses"
            elif key == "train_acc":
                history_key = "train_accuracies"
            elif key == "val_loss":
                history_key = "val_losses"
            elif key == "val_acc":
                history_key = "val_accuracies"

            if history_key in self.training_results:
                self.training_results[history_key].append(value)

        # 更新当前值
        self.training_results.update(update_data)

    def _calculate_gradient_norm(self, model=None):
        """
        计算模型参数的梯度范数

        计算所有模型参数梯度的L2范数，用于监控训练稳定性

        Args:
            model: 神经网络模型，如果为None则尝试使用self.model

        Returns:
            float: 梯度的L2范数

        Example:
            >>> gradient_norm = self._calculate_gradient_norm(model)
        """
        if model is None:
            if hasattr(self, "model"):
                model = self.model
            else:
                return 0.0

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1.0 / 2)

    def _log_epoch_results(self, epoch):
        """
        记录当前轮次的训练结果

        将训练和验证指标输出到日志中

        Args:
            epoch: 当前轮次编号

        Example:
            >>> self._log_epoch_results(epoch)
        """
        logger.info(
            f"训练和验证指标 - Epoch {epoch+1}/{config.EPOCHS}: "
            f"train_loss={self.training_results['train_loss']:.4f}, "
            f"train_acc={self.training_results['train_acc']:.4f}, "
            f"val_loss={self.training_results['val_loss']:.4f}, "
            f"val_acc={self.training_results['val_acc']:.4f}"
        )

    def _validate_input_parameters(self, data_dir, cipher_name):
        """
        验证输入参数的有效性

        检查训练所需的参数是否符合要求

        Args:
            data_dir (str): 数据目录路径
            cipher_name (str): 密码算法名称

        Raises:
            ValueError: 当参数不符合要求时
            FileNotFoundError: 当数据目录不存在时

        Example:
            >>> self._validate_input_parameters("./data", "present")
        """
        # 检查参数是否为空
        if data_dir is None or cipher_name is None:
            raise ValueError("必须提供data_dir和cipher_name参数")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        if not isinstance(cipher_name, str) or len(cipher_name.strip()) == 0:
            raise ValueError("cipher_name必须是非空字符串")

    def _setup_callbacks(self, callbacks_config):
        """
        设置回调管理器和相关回调函数

        创建并配置回调管理器，添加绘图和早停回调函数

        Returns:
            CallbackManager: 配置好的回调管理器

        Example:
            >>> callback_manager = self._setup_callbacks()
        """
        # 初始化回调管理器并设置训练器实例引用
        callback_manager = CallbackManager()
        callback_manager.set_trainer(self)

        # 添加绘图回调函数
        plotting_callback = PlottingCallback(
            save_dir=config.PLOT_SAVE_DIR,
            rounds=callbacks_config["rounds"],
            experiment_name=callbacks_config["data_prefix"],
            cipher_name=callbacks_config["cipher_name"],
        )
        callback_manager.add_callback(plotting_callback)

        # 添加早停回调函数
        early_stopping_callback = EarlyStoppingCallback(patience=30)
        callback_manager.add_callback(early_stopping_callback)

        return callback_manager

    def train_model(self, data_dir, cipher_name, rounds=4, data_prefix="exp1"):
        """
        训练模型

        执行完整的模型训练流程，包括数据准备、模型初始化、训练循环和测试评估

        Args:
            data_dir (str): 数据目录路径
            cipher_name (str): 密码算法名称，用于构建文件路径
            rounds (int): 密码轮数，默认为4
            data_prefix (str): 数据路径前缀，默认为"exp1"，可设置为"exp2"等

        Returns:
            dict: 包含训练和测试结果的字典

        Raises:
            ValueError: 当输入参数不符合要求时
            RuntimeError: 当训练过程中发生错误时
            FileNotFoundError: 当数据文件不存在时

        Example:
            >>> trainer = TrainModel(hidden_dim=128, num_layers=2)
            >>> results = trainer.train_model(
            ...     data_dir="./data", cipher_name="present"
            ... )
            >>> # 使用exp2数据集
            >>> results = trainer.train_model(
            ...     data_dir="./data", cipher_name="present", 
                    data_prefix="exp2"
            ... )
        """
        # 参数验证
        self._validate_input_parameters(data_dir, cipher_name)

        logger.info(f"使用了{self.device}进行训练")
        if str(self.device) == "cuda":
            logger.info(f"GPU 型号是: {torch.cuda.get_device_name(0)}")

        try:
            # 准备数据
            (X_train, y_train, X_val, y_val,
             X_test, y_test) = self._prepare_data(
                data_dir, cipher_name, rounds, data_prefix
            )
            # 创建数据加载器
            train_loader, val_loader = self._create_data_loaders(
                X_train, y_train, X_val, y_val
            )

            # 准备模型、优化器和损失函数配置
            model_config = {
                # 模型配置
                "input_size": X_train.shape[-1],  # 输入特征维度
                "hidden_size": self.train_parameters.get("hidden_dim", 64),
                "num_layers": self.train_parameters.get("num_layers", 2),
                "output_size": y_train.shape[-1],  # 输出维度
                "dropout": self.train_parameters.get("dropout", 0.1),
                # 优化器配置
                "optimizer": self.train_parameters.get("optimizer", "adam"),
                "learning_rate": self.train_parameters.get("lr", 0.001),
                # 损失函数配置
                "criterion": self.train_parameters.get("criterion", "bce"),
            }

            # 初始化模型、优化器和损失函数
            model, optimizer, criterion = self._setup_model(model_config)
            # 确保训练完成后可直接使用当前模型进行测试
            self.model = model

            # 初始化回调管理器
            callbacks_config = {
                "data_dir": data_dir,
                "cipher_name": cipher_name,
                "data_prefix": data_prefix,
                "rounds": rounds
            }
            callback_manager = self._setup_callbacks(callbacks_config)
        except Exception as e:
            logger.error(f"数据准备或模型初始化失败: {str(e)}")
            raise RuntimeError(f"训练初始化失败: {str(e)}") from e

        # 调用训练开始回调
        callback_manager.on_train_begin()

        # 训练循环
        try:
            for epoch in range(config.EPOCHS):
                try:
                    # 执行训练步骤
                    train_loss, train_acc = self._train_epoch(
                        model, train_loader, optimizer, criterion
                    )

                    # 执行验证步骤
                    val_metrics = self._validate_epoch(
                        model, val_loader, criterion
                    )

                    # 更新训练结果
                    self._update_training_results(
                        train_loss, train_acc, val_metrics, optimizer, model
                    )

                    # 记录训练日志
                    self._log_epoch_results(epoch)

                    # 调用回调函数并检查早停
                    callback_results = callback_manager.on_epoch_end(epoch)
                    if callback_results.get("early_stop", False):
                        logger.info(f"在轮次{epoch+1} 处提前停止训练")
                        self.training_results["stopped_early"] = True
                        break
                except Exception as e:
                    logger.error(f"训练轮次 {epoch+1} 失败: {str(e)}")
                    self.training_results["stopped_early"] = True
                    self.training_results["error_message"] = str(e)
                    raise RuntimeError(f"训练在轮次{epoch+1}失败: {str(e)}") from e
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            self.training_results["stopped_early"] = True
        except Exception as e:
            logger.error(f"训练过程中发生严重错误: {str(e)}")
            raise RuntimeError(f"训练失败: {str(e)}") from e

        # 训练结束时调用所有回调函数
        # 回调函数可以直接通过self.training_results访问所有训练数据
        # 在调用on_train_end前补充总轮次
        self.training_results["total_epochs"] = (
            (epoch + 1) if "epoch" in locals() else 0
        )
        callback_manager.on_train_end()

        # 如果存在测试数据，则在训练结束后进行测试集评估
        try:
            if X_test is not None and y_test is not None:
                self.evaluate_on_test_set(self.model, X_test, y_test)
        except Exception as e:
            logger.warning(
                "测试集评估失败: %s",
                str(e)
            )

        # 计算最终训练指标（取最后5个epoch的平均值）
        final_train_acc = (
            np.mean(self.training_results["train_accuracies"][-5:])
            if len(self.training_results["train_accuracies"]) >= 5
            else self.training_results["train_acc"]
        )
        final_val_acc = (
            np.mean(self.training_results["val_accuracies"][-5:])
            if len(self.training_results["val_accuracies"]) >= 5
            else self.training_results["val_acc"]
        )

        logger.info(
            f"最终训练和验证指标 - final_train_acc: {final_train_acc:.4f}, "
            f"final_val_acc: {final_val_acc:.4f}"
        )

        return {
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "training_results": self.training_results,
        }

    def evaluate_on_test_set(self, model, X_test, y_test):
        """
        在测试集上评估模型性能

        对训练完成的模型在独立测试集上进行全面评估

        Args:
            model (torch.nn.Module): 已训练的模型
            X_test (torch.Tensor): 测试集输入数据
            y_test (torch.Tensor): 测试集标签数据

        Returns:
            dict: 包含测试集评估结果的字典

        Example:
            >>> test_results = self.evaluate_on_test_set(model, X_test, y_test)
        """
        # 参数验证
        if model is None or X_test is None or y_test is None:
            raise ValueError("模型和测试数据不能为空")
        if len(X_test) != len(y_test):
            raise ValueError(
                f"测试数据维度不匹配: X_test={len(X_test)}, y_test={len(y_test)}"
            )

        logger.info("开始在测试集上评估模型性能...")

        # 创建测试数据加载器
        test_dataset = TensorDataset(X_test, y_test)
        # 修复：self.train_parameters 是字典，不是可调用对象
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.train_parameters.get("batch_size", 32),
            shuffle=False,
        )

        # 设置模型为评估模式
        model.eval()
        criterion = torch.nn.BCELoss()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                batch_result = self._process_batch(
                    model, inputs, labels, criterion, None, False
                )

                if batch_result["skip"]:
                    logger.warning(
                        f"测试批次 {batch_idx} 出现异常损失值: {batch_result['loss']}"
                    )
                    continue

                total_loss += batch_result["loss"]
                total_correct += batch_result["correct"]
                total_samples += batch_result["total_samples"]
                all_predictions.extend(batch_result["predictions"])
                all_targets.extend(batch_result["targets"])

        if total_samples == 0:
            raise RuntimeError("没有成功处理任何测试批次")

        # 计算成功率指标
        try:
            predictions_tensor = torch.tensor(np.array(all_predictions))
            targets_tensor = torch.tensor(np.array(all_targets))
            bitwise_sr = bitwise_success_rate(
                predictions_tensor, targets_tensor
            )
            log2_sr = log2_success_rate(bitwise_sr)
        except Exception as e:
            logger.warning(f"测试集成功率计算失败: {str(e)}")
            bitwise_sr = log2_sr = 0.0

        # 记录测试结果
        test_results = {
            "test_loss": total_loss / len(test_loader),
            "test_acc": total_correct / total_samples,
            "test_bitwise_success_rate": bitwise_sr,
            "test_log2_success_rate": log2_sr,
        }

        # 记录到训练结果中
        self.training_results.update(test_results)

        # 输出测试结果日志
        logger.info(
            f"测试集评估结果 - test_loss: {test_results['test_loss']:.4f}, "
            f"test_acc: {test_results['test_acc']:.4f}, "
            f"test_bitwise_success_rate: {bitwise_sr:.4f}, "
            f"test_log2_success_rate: {log2_sr:.4f}"
        )

        return test_results

    def test_model(self, model_path=None, test_data_path=None):
        """
        独立的测试集评估方法

        加载已保存的模型或使用当前模型，在指定的测试集上进行评估

        Args:
            model_path (str, optional): 已保存模型的路径
            test_data_path (str, optional): 测试数据路径

        Returns:
            dict: 测试集评估结果字典

        Example:
            >>> results = trainer.test_model()
        """
        logger.info("开始独立测试集评估...")

        # 加载测试数据
        if test_data_path is not None:
            X_test_np = np.load(f"{test_data_path}/plain_texts.npy")
            y_test_np = np.load(f"{test_data_path}/cipher_texts.npy")
            X_test = torch.from_numpy(X_test_np).float().unsqueeze(1)
            y_test = torch.from_numpy(y_test_np).float()
            logger.info(f"从 {test_data_path} 加载测试数据")
        else:
            # 使用默认测试数据
            if not hasattr(self, "data_dir") or \
               not hasattr(self, "cipher_name"):
                raise ValueError(
                    "请提供test_data_path参数，或确保已设置data_dir和cipher_name属性"
                )

            data_prefix = getattr(self, "data_prefix", "exp1")
            test_path = (
                f"{self.data_dir}/{data_prefix}_test_{self.cipher_name}"
            )
            X_test_np = np.load(f"{test_path}/plain_texts.npy")
            y_test_np = np.load(f"{test_path}/cipher_texts.npy")
            X_test = torch.from_numpy(X_test_np).float().unsqueeze(1)
            y_test = torch.from_numpy(y_test_np).float()
            logger.info(f"从默认路径 {test_path} 加载测试数据")

        # 加载或使用模型
        if model_path is not None:
            model = CipherLSTM(
                input_dim=X_test.shape[2],
                hidden_dim=self.train_parameters.get("hidden_dim", 128),
                num_layers=self.train_parameters.get("num_layers", 2),
                output_dim=1 if len(y_test.shape) == 1 else y_test.shape[1],
            )
            model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            model.to(self.device)
            logger.info(f"从 {model_path} 加载模型")
        else:
            if not hasattr(self, "model") or self.model is None:
                raise ValueError("请提供model_path参数或确保已完成模型训练")
            model = self.model
            logger.info("使用当前训练的模型")

        # 进行测试集评估
        test_results = self.evaluate_on_test_set(model, X_test, y_test)

        logger.info("独立测试集评估完成")
        return test_results
