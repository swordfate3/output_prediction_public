"""实验全局配置（纯 JSON 驱动）

[DEL] 删除所有类属性，改为仅通过外部 JSON 文件提供配置；
      统一使用函数接口读取配置，避免在代码中直接访问属性。

[ADD] 支持运行时加载 `configs/config.json`，并提供一组小驼峰式
      函数以读取模型类型、训练轮数、搜索空间、算法注册表等。
"""

from utils.directory_manager import (
    get_data_directory,
    get_plots_directory,
    get_models_directory,
    get_global_directory_manager,
    get_results_directory,
)
from typing import Dict, Any
import os
import sys
import json


class Config:
    def __init__(self):
        """初始化配置实例并加载外部 JSON 配置

        详细描述：不再在代码中定义任何顶层配置常量；所有配置（如模型类型、
        训练轮数、搜索空间、算法注册表、实验配置和最佳超参数等）均由
        `configs/config.json` 提供。本构造函数会查找并解析该 JSON，
        若存在引用占位符（如 `"$BEST_HYPERPARAMETERS"`），将进行解析。

        Raises:
            ValueError: 当未找到配置文件或 JSON 内容不合法时。

        Example:
            >>> cfg = Config()
        """
        # [ADD] 加载运行时 JSON 配置
        path = self.findRuntimeConfigPath()
        if not path:
            raise ValueError("未找到外部 JSON 配置文件 configs/config.json")
        self._cfg = self.loadJsonConfig(path)
        self._resolveBestHyperparamsLink()

    def findRuntimeConfigPath(self) -> str | None:
        """查找运行时 JSON 配置文件路径

        优先级顺序：
        1. 环境变量 `OUTPUT_PREDICTION_CONFIG_PATH`
        2. 当前工作目录 `./configs/config.json` 或 `./config.json`
        3. 脚本所在目录的上级 `../configs/config.json`
        4. 打包可执行目录 `dirname(sys.executable)/configs/config.json`
        5. PyInstaller 一文件模式临时目录 `sys._MEIPASS/configs/config.json`

        Returns:
            str | None: 若找到有效文件，返回其绝对路径；否则返回 None。

        Example:
            >>> Config().findRuntimeConfigPath()
        """
        env_path = os.environ.get("OUTPUT_PREDICTION_CONFIG_PATH")
        if env_path and os.path.isfile(env_path):
            return os.path.abspath(env_path)

        candidates = [
            os.path.join(os.getcwd(), "configs", "config.json"),
            os.path.join(os.getcwd(), "config.json"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")),
        ]

        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, "configs", "config.json"))

        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "configs", "config.json"))

        for p in candidates:
            if os.path.isfile(p):
                return os.path.abspath(p)
        return None

    def loadJsonConfig(self, path: str) -> Dict[str, Any]:
        """加载并解析 JSON 配置文件

        Args:
            path (str): JSON 文件的绝对路径。

        Returns:
            Dict[str, Any]: 解析后的字典对象。

        Raises:
            ValueError: 当文件内容不是合法 JSON 或结构不符合预期时。

        Example:
            >>> Config().loadJsonConfig("./configs/config.json")
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"读取 JSON 配置失败: {path}, 错误: {exc}")

        if not isinstance(data, dict):
            raise ValueError("外部 JSON 配置必须是字典对象")
        return data

    def _resolveBestHyperparamsLink(self) -> None:
        """解析 `EXPERIMENT_CONFIGS.exp2.best_hyperparams` 的占位符链接

        支持在 JSON 中使用字符串占位符 `"$BEST_HYPERPARAMETERS"`，
        以复用当前最优超参数配置，避免在 JSON 中重复拷贝内容。

        Example:
            >>> Config()._resolveBestHyperparamsLink()
        """
        try:
            exp2 = self._cfg.get("EXPERIMENT_CONFIGS", {}).get("exp2", {})
            bh = exp2.get("best_hyperparams")
            if isinstance(bh, str) and bh == "$BEST_HYPERPARAMETERS":
                self._cfg["EXPERIMENT_CONFIGS"]["exp2"]["best_hyperparams"] = self._cfg.get("BEST_HYPERPARAMETERS", {})
        except Exception:
            pass

    # -------------------------------
    # 目录相关函数（替代属性）
    # -------------------------------
    def getDataDirectory(self) -> str:
        """获取数据目录路径

        Returns:
            str: 数据目录的绝对路径。

        Example:
            >>> Config().getDataDirectory()
        """
        return get_data_directory()

    def getModelDirectory(self) -> str:
        """获取模型目录路径

        Returns:
            str: 模型目录的绝对路径。

        Example:
            >>> Config().getModelDirectory()
        """
        return get_models_directory()

    def getResultsDirectory(self) -> str:
        """获取结果目录路径

        Returns:
            str: 结果目录的绝对路径。

        Example:
            >>> Config().getResultsDirectory()
        """
        return get_results_directory()

    def getPlotsDirectory(self) -> str:
        """获取绘图保存目录路径

        Returns:
            str: 绘图目录的绝对路径。

        Example:
            >>> Config().getPlotsDirectory()
        """
        return get_plots_directory()

    def getDirectoryManager(self):
        """获取目录管理器实例

        Returns:
            Any: 目录管理器实例。

        Example:
            >>> Config().getDirectoryManager()
        """
        return get_global_directory_manager()

    def setBaseDirectory(self, base_dir: str):
        """设置基础目录

        Args:
            base_dir (str): 根目录绝对路径。

        Returns:
            Any: 重置后的目录管理器实例。

        Example:
            >>> Config().setBaseDirectory("/tmp/output_prediction")
        """
        from utils.directory_manager import reset_global_directory_manager

        reset_global_directory_manager()
        return get_global_directory_manager(base_dir)

    # -------------------------------
    # JSON 配置读取函数
    # -------------------------------
    def getModelType(self) -> str:
        """获取模型类型（如 'lstm'、'itransformer'、'mamba'）

        Returns:
            str: 模型类型字符串。

        Example:
            >>> Config().getModelType()
        """
        return str(self._cfg.get("MODEL_TYPE", "lstm"))

    def getEpochs(self) -> int:
        """获取训练轮数 epochs

        Returns:
            int: 训练轮数。

        Example:
            >>> Config().getEpochs()
        """
        return int(self._cfg.get("EPOCHS", 200))

    def getBlockSize(self) -> int:
        """获取默认块大小（按位）

        Returns:
            int: 块大小。

        Example:
            >>> Config().getBlockSize()
        """
        return int(self._cfg.get("BLOCK_SIZE", 128))

    def getKeySize(self) -> int:
        """获取默认密钥大小（按位）

        Returns:
            int: 密钥大小。

        Example:
            >>> Config().getKeySize()
        """
        return int(self._cfg.get("KEY_SIZE", 128))

    def getHyperparameterSearchSpace(self) -> Dict[str, Any]:
        """获取超参数搜索空间配置

        Returns:
            Dict[str, Any]: 用于 Optuna 的超参搜索空间。

        Example:
            >>> Config().getHyperparameterSearchSpace()
        """
        space = self._cfg.get("HYPERPARAMETER_SEARCH_SPACE", {})
        return dict(space)

    def getBestHyperparameters(self, cipher_name: str) -> Dict[str, Any]:
        """获取指定密码算法的最佳超参数配置

        Args:
            cipher_name (str): 密码算法名称。

        Returns:
            Dict[str, Any]: 最佳超参数字典。

        Raises:
            ValueError: 当给定算法无配置时抛出。

        Example:
            >>> Config().getBestHyperparameters("present")
        """
        all_hp = self._cfg.get("BEST_HYPERPARAMETERS", {})
        if cipher_name not in all_hp:
            raise ValueError(f"未找到 {cipher_name} 的最佳超参数配置")
        return dict(all_hp[cipher_name])

    def getExperimentConfig(self, experiment_name: str) -> Dict[str, Any]:
        """获取指定实验的配置参数

        Args:
            experiment_name (str): 实验名称（exp1、exp2、exp3）。

        Returns:
            Dict[str, Any]: 实验配置字典。

        Raises:
            ValueError: 当实验名无配置时抛出。

        Example:
            >>> Config().getExperimentConfig("exp1")
        """
        exps = self._cfg.get("EXPERIMENT_CONFIGS", {})
        if experiment_name not in exps:
            raise ValueError(f"未找到实验 {experiment_name} 的配置")
        return dict(exps[experiment_name])

    def getCipherRegistry(self) -> Dict[str, Any]:
        """获取密码算法注册表配置

        Returns:
            Dict[str, Any]: 包含算法名称到模块/类路径的映射。

        Example:
            >>> Config().getCipherRegistry()
        """
        reg = self._cfg.get("CIPHER_REGISTRY", {})
        return dict(reg)

    def getSupportedCiphers(self) -> list:
        """获取支持的密码算法列表

        Returns:
            list: 算法名称列表。

        Example:
            >>> Config().getSupportedCiphers()
        """
        return list(self.getCipherRegistry().keys())

    def validateCipherName(self, cipher_name: str) -> bool:
        """验证密码算法名称是否有效

        Args:
            cipher_name (str): 算法名称。

        Returns:
            bool: 有效返回 True，否则 False。

        Example:
            >>> Config().validateCipherName("present")
        """
        return cipher_name in self.getCipherRegistry()

    def getCipherInstance(
        self,
        cipher_name: str,
        rounds: int = 4,
        block_size: int | None = None,
        key_size: int | None = None,
    ):
        """根据密码算法名称创建密码实例（支持覆盖尺寸）

        Args:
            cipher_name (str): 算法名称，需存在于注册表。
            rounds (int): 算法轮数。
            block_size (int | None): 覆盖块大小（位）。
            key_size (int | None): 覆盖密钥大小（位）。

        Returns:
            object: 算法实例。

        Raises:
            ValueError: 算法名无效或尺寸非法。
            ImportError: 模块导入失败。
            AttributeError: 类不存在或不支持属性覆盖。

        Example:
            >>> Config().getCipherInstance("present", 4)
        """
        registry = self.getCipherRegistry()
        if cipher_name not in registry:
            raise ValueError(f"不支持的密码算法: {cipher_name}")

        cipher_config = registry[cipher_name]
        module_path = cipher_config["module_path"]
        class_name = cipher_config["class_name"]

        try:
            import importlib
            # [ADD] 引入 inspect 以自适应不同算法的构造函数签名
            import inspect

            module = importlib.import_module(module_path)
            cipher_class = getattr(module, class_name)
            # [MOD] 自适应传参：仅向构造函数传递其支持的参数，避免不兼容错误（例如 Grain128a 不支持 block_size/key_size）
            init_sig = inspect.signature(cipher_class.__init__)
            init_params = init_sig.parameters
            ctor_kwargs = {"rounds": rounds}
            if block_size is not None and "block_size" in init_params:
                ctor_kwargs["block_size"] = int(block_size)
            if key_size is not None and "key_size" in init_params:
                ctor_kwargs["key_size"] = int(key_size)
            instance = cipher_class(**ctor_kwargs)

            # [MOD] 若构造函数不支持尺寸参数，则在实例化后按需覆盖属性（保持向后兼容）
            if block_size is not None and "block_size" not in init_params:
                bs = int(block_size)
                if bs <= 0:
                    raise ValueError(f"block_size 必须为正整数，当前值: {block_size}")
                if not hasattr(instance, "block_size"):
                    raise AttributeError(f"算法实例不支持设置 block_size 属性: {class_name}")
                instance.block_size = bs

            if key_size is not None and "key_size" not in init_params:
                ks = int(key_size)
                if ks <= 0:
                    raise ValueError(f"key_size 必须为正整数，当前值: {key_size}")
                if not hasattr(instance, "key_size"):
                    raise AttributeError(f"算法实例不支持设置 key_size 属性: {class_name}")
                instance.key_size = ks

            return instance
        except ImportError as e:
            raise ImportError(f"无法导入模块 {module_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"模块 {module_path} 中不存在类 {class_name}: {e}")

    def getDataPath(self, experiment_name: str, cipher_name: str, data_type: str = "train") -> str:
        """生成实验数据路径

        Args:
            experiment_name (str): 实验名称。
            cipher_name (str): 密码算法名称。
            data_type (str): 数据类型（'train' 或 'test'）。

        Returns:
            str: 数据路径。

        Example:
            >>> Config().getDataPath("exp1", "present", "train")
        """
        import os

        exp_config = self.getExperimentConfig(experiment_name)
        data_prefix = exp_config["data_prefix"]
        return os.path.join(self.getDataDirectory(), f"{data_prefix}_{data_type}_{cipher_name}")

    def getCompleteConfig(self, experiment_name: str, cipher_name: str, rounds: int = 4) -> Dict[str, Any]:
        """获取完整的实验配置（实验+算法+超参）

        Args:
            experiment_name (str): 实验名称。
            cipher_name (str): 密码算法名称。
            rounds (int): 轮数。

        Returns:
            Dict[str, Any]: 完整配置字典。

        Example:
            >>> Config().getCompleteConfig("exp1", "present", 4)
        """
        exp_config = self.getExperimentConfig(experiment_name)
        cipher_config = self.getCipherRegistry().get(cipher_name, {})
        hyperparams = self.getBestHyperparameters(cipher_name)
        return {
            "experiment": exp_config,
            "cipher": {"name": cipher_name, "rounds": rounds, "class_info": cipher_config},
            "hyperparameters": hyperparams,
            "data_paths": {
                "train": self.getDataPath(experiment_name, cipher_name, "train"),
                "test": self.getDataPath(experiment_name, cipher_name, "test"),
            },
        }


# 实例化配置对象
config = Config()
