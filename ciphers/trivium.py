import numpy as np

# [ADD] 新增: 兼容直接运行与包运行的导入方式，确保 Trivium 与项目一致
try:
    from .base_cipher import BaseCipher  # type: ignore
except Exception:  # [ADD] 回退导入，支持在无包上下文下直接运行文件
    from ciphers.base_cipher import BaseCipher  # type: ignore


class Trivium(BaseCipher):
    """
    Trivium 流密码的核心实现（ECRYPT 推荐算法），生成密钥流并进行异或加解密。

    - 接口遵循项目约定：位数组输入/输出（np.uint8，元素为 0/1）
    - 明文长度固定为 128 比特（`block_size=128`）用于与现有数据生成与训练流程对齐
    - 密钥长度为 80 比特（`key_size=80`），IV 也为 80 比特
    - 内部状态：三条移位寄存器，长度分别为 93、84、111
    - 预热（初始化）固定 1152 次，不输出密钥流，仅混合更新状态
    - 工作阶段：每步生成一个密钥流比特 `z`，并按规范更新三条寄存器

    说明：该实现对齐 Trivium 的公开规范（0 基索引），便于在黑盒密码分析与输出预测实验中使用。
    """

    def __init__(self, rounds: int = 1152, iv: np.ndarray | None = None):
        """
        初始化 Trivium 密码实例。

        Args:
            rounds (int): 预热步数（规范为 1152），默认 1152
            iv (np.ndarray | None): 80 比特的 IV；若为 None，则随机生成

        Returns:
            None

        Raises:
            ValueError: 当 IV 长度不是 80 比特时抛出异常

        Example:
            >>> cipher = Trivium()
            >>> key = np.random.randint(0, 2, size=cipher.key_size, dtype=np.uint8)
            >>> pt = np.random.randint(0, 2, size=cipher.block_size, dtype=np.uint8)
            >>> ct = cipher.encrypt(pt, key)
        """
        super().__init__(block_size=128, key_size=80)
        # [DEL] 移除构造函数中的调试打印，避免在库场景产生副作用
        # print(f"使用的轮数: {rounds}")
        self.rounds = int(rounds)
        if iv is None:
            iv_bits = self.generateRandomBits(80)
        else:
            iv_bits = np.array(iv, dtype=np.uint8).flatten()
            if iv_bits.size != 80:
                raise ValueError("Trivium 的 IV 需为80比特")
        self._iv = iv_bits

        # [ADD] 运行态寄存器：三条移位寄存器，长度分别为 93、84、111
        self._s1 = np.zeros(93, dtype=np.uint8)
        self._s2 = np.zeros(84, dtype=np.uint8)
        self._s3 = np.zeros(111, dtype=np.uint8)

    def _loadState(self, key_bits: np.ndarray) -> None:
        """
        加载初始状态到三条寄存器（93、84、111）。

        详细描述：
        - s1[0..79] = key[0..79]；s1[80..92] = 0
        - s2[0..79] = iv[0..79]；s2[80..83] = 0
        - s3[0..107] = 0；s3[108..110] = 1（最后三个比特全为 1）

        Args:
            key_bits (np.ndarray): 80 位密钥（np.uint8，元素为 0/1）

        Returns:
            None

        Raises:
            ValueError: 当密钥长度不是 80 位时抛出异常

        Example:
            >>> cipher = Trivium()
            >>> key = np.zeros(80, dtype=np.uint8)
            >>> cipher._loadState(key)
        """
        k = np.array(key_bits, dtype=np.uint8).flatten()
        if k.size != 80:
            raise ValueError("Trivium 需要80位密钥")

        # s1: key || 13*0
        self._s1[:] = 0
        self._s1[:80] = k
        # s2: iv || 4*0
        self._s2[:] = 0
        self._s2[:80] = self._iv
        # s3: 108*0 || 3*1
        self._s3[:] = 0
        self._s3[108] = 1
        self._s3[109] = 1
        self._s3[110] = 1

    @staticmethod
    def _shiftLeft(arr: np.ndarray, new_bit: int) -> np.ndarray:
        """
        左移一位并在最低位插入新比特。

        Args:
            arr (np.ndarray): 输入比特数组（np.uint8，元素为 0/1）
            new_bit (int): 新插入的最低位比特（0 或 1）

        Returns:
            np.ndarray: 左移后的新数组

        Example:
            >>> Trivium._shiftLeft(np.array([1,0,1], dtype=np.uint8), 1)
            array([0, 1, 1], dtype=uint8)
        """
        out = np.empty_like(arr)
        out[:-1] = arr[1:]
        out[-1] = int(new_bit) & 1
        return out

    def _warmup(self, mix_rounds: int) -> None:
        """
        预热阶段：固定执行 `mix_rounds` 次（规范为 1152），不输出密钥流。

        Args:
            mix_rounds (int): 预热步数

        Returns:
            None

        Example:
            >>> cipher = Trivium()
            >>> cipher._warmup(1152)
        """
        steps = int(mix_rounds)
        for _ in range(steps):
            # taps（0基索引）：
            t1 = int(self._s1[65]) ^ int(self._s1[92])
            t2 = int(self._s2[68]) ^ int(self._s2[83])
            t3 = int(self._s3[65]) ^ int(self._s3[110])

            t1p = t1 ^ (int(self._s1[90]) & int(self._s1[91])) ^ int(self._s2[77])
            t2p = t2 ^ (int(self._s2[81]) & int(self._s2[82])) ^ int(self._s3[86])
            t3p = t3 ^ (int(self._s3[108]) & int(self._s3[109])) ^ int(self._s1[68])

            # 更新寄存器：s1<-t3', s2<-t1', s3<-t2'
            self._s1 = self._shiftLeft(self._s1, t3p)
            self._s2 = self._shiftLeft(self._s2, t1p)
            self._s3 = self._shiftLeft(self._s3, t2p)

    def _keystream(self, n_bits: int, key_bits: np.ndarray) -> np.ndarray:
        """
        生成指定长度的密钥流。

        详细描述：加载状态并执行固定 1152 次预热，然后在工作阶段对寄存器
        进行标准更新，逐位产出密钥流。

        Args:
            n_bits (int): 密钥流长度（比特数）
            key_bits (np.ndarray): 80 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 长度为 n_bits 的密钥流（np.uint8，元素为 0/1）

        Example:
            >>> cipher = Trivium()
            >>> key = np.random.randint(0, 2, size=80, dtype=np.uint8)
            >>> ks = cipher._keystream(128, key)
        """
        self._loadState(key_bits)
        self._warmup(self.rounds)
        ks = np.zeros(int(n_bits), dtype=np.uint8)
        for i in range(int(n_bits)):
            # taps（0基索引）
            t1 = int(self._s1[65]) ^ int(self._s1[92])
            t2 = int(self._s2[68]) ^ int(self._s2[83])
            t3 = int(self._s3[65]) ^ int(self._s3[110])

            z = (t1 ^ t2 ^ t3) & 1
            ks[i] = z

            t1p = t1 ^ (int(self._s1[90]) & int(self._s1[91])) ^ int(self._s2[77])
            t2p = t2 ^ (int(self._s2[81]) & int(self._s2[82])) ^ int(self._s3[86])
            t3p = t3 ^ (int(self._s3[108]) & int(self._s3[109])) ^ int(self._s1[68])

            # 更新寄存器：s1<-t3', s2<-t1', s3<-t2'
            self._s1 = self._shiftLeft(self._s1, t3p)
            self._s2 = self._shiftLeft(self._s2, t1p)
            self._s3 = self._shiftLeft(self._s3, t2p)
        return ks

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对明文进行加密（逐位异或）。

        详细描述：生成与明文等长的密钥流，然后逐位异或得到密文。

        Args:
            plaintext (np.ndarray): 128 位明文（np.uint8，元素为 0/1）
            key (np.ndarray): 80 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位密文（np.uint8，元素为 0/1）

        Raises:
            ValueError: 当输入长度不匹配时抛出

        Example:
            >>> cipher = Trivium()
            >>> key = np.random.randint(0, 2, size=cipher.key_size, dtype=np.uint8)
            >>> pt = np.random.randint(0, 2, size=cipher.block_size, dtype=np.uint8)
            >>> ct = cipher.encrypt(pt, key)
        """
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if pt.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"Trivium 仅支持{self.block_size}位明文与{self.key_size}位密钥"
            )
        ks = self._keystream(pt.size, k)
        return np.bitwise_xor(pt, ks).astype(np.uint8)

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对密文进行解密（与加密相同，逐位异或）。

        详细描述：生成与密文等长的密钥流，然后逐位异或得到明文。

        Args:
            ciphertext (np.ndarray): 128 位密文（np.uint8，元素为 0/1）
            key (np.ndarray): 80 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位明文（np.uint8，元素为 0/1）

        Raises:
            ValueError: 当输入长度不匹配时抛出

        Example:
            >>> cipher = Trivium()
            >>> key = np.random.randint(0, 2, size=cipher.key_size, dtype=np.uint8)
            >>> ct = np.random.randint(0, 2, size=cipher.block_size, dtype=np.uint8)
            >>> pt = cipher.decrypt(ct, key)
        """
        ct = np.array(ciphertext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if ct.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"Trivium 仅支持{self.block_size}位密文与{self.key_size}位密钥"
            )
        ks = self._keystream(ct.size, k)
        return np.bitwise_xor(ct, ks).astype(np.uint8)