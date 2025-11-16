import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)


class BaseCipher:
    """分组密码基类，定义通用接口"""

    def __init__(self, block_size, key_size):
        self.block_size = block_size  # 块大小（比特）
        self.key_size = key_size  # 密钥大小（比特）
        self.rounds = 0  # 轮数（由子类设置）
        self.S_BOX = None  # S盒（子类实现）
        # [ADD] 新增：统一的随机数生成器，保证密钥与明文生成一致可复现
        # 说明：在构造函数中创建一次生成器，避免各函数重复实例化
        self._rng = np.random.default_rng(seed=42)
        # [DEL] 删除：generate_key 与 generate_plaintext 两个方法
        # 原因：统一改为使用 generateRandomBits(length) 生成随机位，
        # 通过传入 key_size 或 block_size 即可获得密钥/明文。

    def encrypt(self, plaintext, key):
        """加密函数，由子类实现"""
        raise NotImplementedError

    def decrypt(self, ciphertext, key):
        """解密函数，由子类实现"""
        raise NotImplementedError

    def generateRandomBits(self, length: int) -> np.ndarray:
        """
        生成统一的随机位数组

        详细描述函数的功能和用途
        使用统一的随机数生成器生成指定长度的 0/1 位数组，
        供密钥与明文生成复用，确保风格一致与可复现。

        Args:
            length (int): 位数组长度（比特数，必须为正整数）

        Returns:
            np.ndarray: 长度为 length 的位数组，元素为 0/1（dtype=np.uint8）

        Raises:
            ValueError: 当 length 小于或等于 0 时抛出异常

        Example:
            >>> base = BaseCipher(block_size=8, key_size=8)
            >>> bits = base.generateRandomBits(4)
            >>> bits.size
            4
        """
        # [ADD] 新增：统一的随机位生成逻辑
        if int(length) <= 0:
            raise ValueError("长度必须为正整数")
        size = (int(length),)
        return self._rng.integers(0, 2, size=size, dtype=np.uint8)
