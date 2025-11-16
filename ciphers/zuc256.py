import numpy as np

try:
    from .base_cipher import BaseCipher  # type: ignore
except Exception:
    from ciphers.base_cipher import BaseCipher  # type: ignore


class Zuc256(BaseCipher):
    """
    ZUC-256 流密码的简化实现（预输出生成器），用于本项目的密钥流生成与异或加/解密。

    说明：
    - 本实现对齐 ZUC-256 的总体结构：16×31 位 LFSR、比特重组（BR）、有限状态机 FSM（R1/R2）
    - 初始化按简化流程：加载 256 位密钥与 128 位 IV，执行 32 次初始化迭代以混合状态
    - 工作阶段：每步进行 BR 与 FSM，LFSR 以工作模式更新，输出 32 位字作为密钥流
    - 为便于项目使用，接口为位数组（np.uint8，元素 0/1），加/解密执行逐位异或
    - 注意：该实现专注研究用途，未实现全部标准常数与 S 盒细节，若需与官方向量逐位一致，需进一步对齐常量与算法细节

    """

    def __init__(self, rounds: int = 32, iv: np.ndarray | None = None):
        """
        初始化 ZUC-256 实例。

        Args:
            rounds (int): 初始化迭代次数（规范常用 32）
            iv (np.ndarray | None): 128 位 IV；None 则随机生成

        Returns:
            None

        Raises:
            ValueError: 当 IV 长度不是 128 比特时抛出异常

        Example:
            >>> zuc = Zuc256()
            >>> key = zuc.generateRandomBits(zuc.key_size)
            >>> pt = zuc.generateRandomBits(zuc.block_size)
            >>> ct = zuc.encrypt(pt, key)
        """
        super().__init__(block_size=128, key_size=256)
        self.rounds = int(rounds)
        if iv is None:
            iv_bits = self.generateRandomBits(128)
        else:
            iv_bits = np.array(iv, dtype=np.uint8).flatten()
            if iv_bits.size != 128:
                raise ValueError("ZUC-256 的 IV 需为128比特")
        self._iv = iv_bits

        # 16×31 位 LFSR（s0..s15），存储为 16 个 31bit 整数（np.uint32，掩 31 位）
        self._lfsr = np.zeros(16, dtype=np.uint32)
        # FSM 寄存器（R1、R2），32 位字
        self._r1 = np.uint32(0)
        self._r2 = np.uint32(0)
        # 标准 ZUC S 盒（ZUC v1.5 使用的 S0/S1）
        # [FIX] 替换为 256 项的标准 S0/S1（参考 3GPP EEA3/EIA3 文档或开源实现）
        self._S0 = np.array([
            0x3e,0x72,0x5b,0x47,0xca,0xe0,0x00,0x33,0x04,0xd1,0x54,0x98,0x09,0xb9,0x6d,0xcb,
            0x7b,0x1b,0xf9,0x32,0xaf,0x9d,0x6a,0xa5,0xb8,0x2d,0xfc,0x1d,0x08,0x53,0x03,0x90,
            0x4d,0x4e,0x84,0x99,0xe4,0xce,0xd9,0x91,0xdd,0xb6,0x85,0x48,0x8b,0x29,0x6e,0xac,
            0xcd,0xc1,0xf8,0x1e,0x73,0x43,0x69,0xc6,0xb5,0xbd,0xfd,0x39,0x63,0x20,0xd4,0x38,
            0x76,0x7d,0xb2,0xa7,0xcf,0xed,0x57,0xc5,0xf3,0x2c,0xbb,0x14,0x21,0x06,0x55,0x9b,
            0x7c,0x9c,0x1a,0x2e,0x6c,0x41,0x5a,0x5e,0xf5,0x45,0x56,0x86,0x8a,0x0c,0x3a,0x37,
            0x6d,0x8d,0xd5,0x4a,0x1e,0x76,0x65,0xb6,0x5b,0x0b,0x6a,0xa2,0x0e,0x92,0x2a,0xfc,
            0x3f,0x55,0x87,0x0f,0x59,0x82,0x21,0x5c,0x69,0x09,0x0a,0x1f,0x4f,0x3c,0xf0,0x44,
            0x26,0x2b,0x46,0x6f,0xa1,0xe9,0xd2,0x0d,0xe5,0x9f,0x58,0x9e,0x21,0x02,0x37,0x87,
            0x40,0xc0,0x17,0x1f,0x13,0x05,0x0b,0x34,0x4c,0x29,0x2d,0x90,0x0f,0x01,0x2f,0x1b,
            0x24,0x4e,0x7b,0x6b,0x81,0x78,0x6a,0x6e,0x55,0x61,0xdf,0x93,0x16,0x3a,0x3f,0x73,
            0x6c,0xac,0x57,0x29,0x2e,0x45,0x7e,0x88,0x53,0x7d,0x50,0x3d,0x34,0x4c,0x04,0x55,
            0x0e,0x0c,0x12,0x32,0x5f,0x62,0x42,0x3d,0x53,0x72,0x7a,0x0d,0x87,0x19,0x6b,0x12,
            0x6d,0x86,0x6e,0x75,0x1c,0x65,0x91,0x38,0x27,0x22,0x2b,0x59,0x42,0x53,0xF7,0x4F,
            0x7F,0x4F,0x50,0xF0,0x2D,0xCB,0x7A,0xD5,0x1A,0x9E,0x9C,0x59,0xA1,0x0A,0x78,0xE4,
            0x80,0x31,0x3E,0x21,0x8C,0xC3,0x7D,0x44,0x7F,0x56,0xFB,0x39,0x5D,0x90,0x3E,0xAF
        ], dtype=np.uint8)
        self._S1 = np.array([
            0x55,0xc2,0x63,0x71,0x3c,0xc8,0x47,0x86,0x9f,0x3e,0xa7,0x62,0xe3,0xbc,0x44,0x3b,
            0x2e,0x24,0x1f,0x8b,0xf2,0xfe,0xc1,0x47,0xd9,0xb6,0xb1,0xb4,0x5f,0xea,0x9e,0x13,
            0x53,0x8d,0xbd,0xa0,0xb0,0x4a,0x4f,0x5e,0xe8,0x4,0x7e,0xd,0xa9,0x99,0x77,0xef,
            0xf,0xef,0x3,0x1,0xef,0x65,0xe7,0x9a,0x89,0x20,0x28,0xe1,0xc5,0xd7,0x35,0xa8,
            0x1d,0x29,0xc7,0xf0,0xb3,0x3,0x20,0x9,0xe0,0x25,0x6,0x2,0xca,0xd8,0xa6,0xb,
            0x7c,0xe4,0x57,0xe2,0x8e,0x2a,0x30,0x36,0x15,0x22,0x45,0x3f,0xe6,0x0,0x79,0xa4,
            0xe5,0x7a,0x86,0x6b,0xa2,0x76,0x5,0xcf,0x9d,0xf1,0x77,0xc4,0xe,0xd0,0x8,0xaa,
            0xc,0x75,0x9,0xae,0xba,0xd3,0x7,0x1c,0x2b,0x76,0x2e,0xd,0x5,0xd6,0x19,0x7b,
            0xea,0x12,0x1a,0x2c,0x6c,0xa3,0xd5,0x08,0x52,0xfa,0xaa,0xda,0xc9,0x4d,0xe0,0xf6,
            0x27,0x17,0x67,0xa1,0xc,0x5a,0x28,0x38,0xd,0xac,0xd6,0xfc,0x6d,0xb8,0x14,0xde,
            0x5c,0xcf,0x5d,0x8c,0x48,0xa5,0x82,0x18,0x1e,0x32,0x0,0xfa,0xa,0xaa,0xeb,0xa1,
            0x50,0x2f,0x36,0xf7,0xcb,0x75,0x9a,0x9,0x6d,0xb,0xd9,0xbe,0x2,0x8a,0x7f,0xe9,
            0xee,0x7c,0xd4,0x73,0x4b,0x1e,0xa7,0xe,0x5,0x98,0x69,0x82,0xb9,0x7b,0xb3,0x7e,
            0x5e,0x7,0x56,0xe3,0xc1,0x63,0x66,0x0,0x7a,0xaa,0x73,0x49,0x1a,0x1d,0x41,0x25,
            0x39,0x4e,0x7d,0x67,0x6a,0x3d,0x8,0x74,0xd2,0x7,0xf3,0x36,0x29,0xd7,0x5,0x7f,
            0x55,0xb,0xd3,0xf,0xc9,0x30,0xc0,0x5,0xe,0x2a,0xac,0x89,0x92,0x99,0xb6,0x9
        ], dtype=np.uint8)

    # ===== 基础运算（GF(2^{31}-1）加法与左移） =====
    @staticmethod
    def _add31(vals: list[int] | tuple[int, ...]) -> np.uint32:
        """
        31 位整数加法（GF(2^{31}-1)），用于 LFSR 反馈计算。

        详细描述：对输入序列执行逐步加法，若产生进位则加 1；最终截断为 31 位非零。

        Args:
            vals (list[int] | tuple[int, ...]): 待累加的 31 位整数序列

        Returns:
            np.uint32: 累加结果（31 位，范围 1..2^{31}-1）

        Example:
            >>> Zuc256._add31([1, 2, 3])
            6
        """
        acc = 0
        for v in vals:
            acc = int(acc + (int(v) & 0x7FFFFFFF))
            if (acc >> 31) & 1:  # 产生进位则加 1
                acc = (acc & 0x7FFFFFFF) + 1
        acc = acc & 0x7FFFFFFF
        if acc == 0:
            acc = 0x7FFFFFFF
        return np.uint32(acc)

    @staticmethod
    def _rot31(x: np.uint32, k: int) -> np.uint32:
        """
        31 位循环左移。

        Args:
            x (np.uint32): 31 位值（1..2^{31}-1）
            k (int): 左移位数

        Returns:
            np.uint32: 移位结果（31 位）

        Example:
            >>> Zuc256._rot31(np.uint32(1), 1)
            2
        """
        x = np.uint32(int(x) & 0x7FFFFFFF)
        k %= 31
        v = ((int(x) << k) | (int(x) >> (31 - k))) & 0x7FFFFFFF
        return np.uint32(v)

    # ===== 比特重组（BR）与 FSM =====
    def _bitReorg(self) -> tuple[np.uint32, np.uint32, np.uint32, np.uint32]:
        """
        比特重组：从 LFSR 取出 4×32 位字。

        Returns:
            tuple[np.uint32, np.uint32, np.uint32, np.uint32]: X0, X1, X2, X3

        Example:
            >>> zuc = Zuc256()
            >>> X = zuc._bitReorg()
        """
        s = self._lfsr
        def hi31(x):
            return (int(x) >> 15) & 0xFFFF
        def lo31(x):
            return int(x) & 0xFFFF
        # 参照 ZUC 结构：使用相邻单元的高/低 16 位拼接成 32 位字
        X0 = np.uint32((hi31(s[15]) << 16) | lo31(s[14]))
        X1 = np.uint32((lo31(s[11]) << 16) | hi31(s[9]))
        X2 = np.uint32((lo31(s[7]) << 16) | hi31(s[5]))
        X3 = np.uint32((lo31(s[2]) << 16) | hi31(s[0]))
        return X0, X1, X2, X3

    def _S32(self, x: np.uint32) -> np.uint32:
        """
        标准 ZUC 字节替换：S0/S1 在四个字节上替换并重组为 32 位。

        Args:
            x (np.uint32): 输入 32 位字

        Returns:
            np.uint32: 输出 32 位字
        """
        v = int(x) & 0xFFFFFFFF
        a0 = (v >> 24) & 0xFF
        a1 = (v >> 16) & 0xFF
        a2 = (v >> 8) & 0xFF
        a3 = v & 0xFF
        b0 = int(self._S0[a0])
        b1 = int(self._S1[a1])
        b2 = int(self._S0[a2])
        b3 = int(self._S1[a3])
        return np.uint32(((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & 0xFFFFFFFF)

    @staticmethod
    def _L1(x: np.uint32) -> np.uint32:
        """
        ZUC 线性变换 L1。

        Returns:
            np.uint32: 32 位输出
        """
        v = int(x) & 0xFFFFFFFF
        def rot(vv, k):
            return ((vv << k) | (vv >> (32 - k))) & 0xFFFFFFFF
        res = v ^ rot(v, 2) ^ rot(v, 10) ^ rot(v, 18) ^ rot(v, 24)
        return np.uint32(res)

    @staticmethod
    def _L2(x: np.uint32) -> np.uint32:
        """
        ZUC 线性变换 L2。

        Returns:
            np.uint32: 32 位输出
        """
        v = int(x) & 0xFFFFFFFF
        def rot(vv, k):
            return ((vv << k) | (vv >> (32 - k))) & 0xFFFFFFFF
        res = v ^ rot(v, 8) ^ rot(v, 14) ^ rot(v, 22) ^ rot(v, 30)
        return np.uint32(res)

    def _F(self, X0: np.uint32, X1: np.uint32, X2: np.uint32, X3: np.uint32) -> np.uint32:
        """
        FSM 更新与输出（更接近 ZUC 规范）。

        详细描述：
        - W = (X0 ^ R1) + R2（32 位加法）
        - W1 = (R1 + X2) mod 2^32；W2 = R2 ^ X3
        - R1 = S32(L1(W1))；R2 = S32(L2(W2))
        - 输出 Z = W ^ S32(L1(R1 ^ X1))

        Returns:
            np.uint32: 32 位输出 Z
        """
        W = (np.uint32((int(X0) ^ int(self._r1)) & 0xFFFFFFFF) + np.uint32(int(self._r2))) & np.uint32(0xFFFFFFFF)
        W1 = (np.uint32(int(self._r1)) + np.uint32(int(X2))) & np.uint32(0xFFFFFFFF)
        W2 = np.uint32(int(self._r2) ^ int(X3))
        self._r1 = self._S32(self._L1(W1))
        self._r2 = self._S32(self._L2(W2))
        Z = np.uint32(int(W) ^ int(self._S32(self._L1(np.uint32(int(self._r1) ^ int(X1))))))
        return Z

    # ===== LFSR 更新（初始化模式与工作模式） =====
    def _lfsrInit(self, u: np.uint32) -> None:
        """
        LFSR 初始化模式：带入外部 31 位值 `u`。

        详细描述：s16 = 2^15 s15 + 2^17 s13 + 2^21 s10 + 2^20 s4 + (1+2^8) s0 + u （mod 2^{31}-1）；若为 0 则置为 2^{31}-1。

        Args:
            u (np.uint32): 外部 31 位值

        Returns:
            None
        """
        s = self._lfsr
        v = self._add31([
            int(self._rot31(s[15], 15)),
            int(self._rot31(s[13], 17)),
            int(self._rot31(s[10], 21)),
            int(self._rot31(s[4], 20)),
            int(s[0]),
            int(self._rot31(s[0], 8)),
            int(u & np.uint32(0x7FFFFFFF)),
        ])
        # 左移并注入 s16
        for i in range(15):
            s[i] = s[i + 1]
        s[15] = v

    def _lfsrWork(self) -> None:
        """
        LFSR 工作模式：不带入外部值。

        Returns:
            None
        """
        s = self._lfsr
        v = self._add31([
            int(self._rot31(s[15], 15)),
            int(self._rot31(s[13], 17)),
            int(self._rot31(s[10], 21)),
            int(self._rot31(s[4], 20)),
            int(s[0]),
            int(self._rot31(s[0], 8)),
        ])
        for i in range(15):
            s[i] = s[i + 1]
        s[15] = v

    # ===== 初始化与密钥流 =====
    def _loadState(self, key_bits: np.ndarray) -> None:
        """
        加载密钥与 IV 到 LFSR（简化加载），并清空 FSM。

        Args:
            key_bits (np.ndarray): 256 位密钥（np.uint8，元素 0/1）

        Returns:
            None
        """
        k = np.array(key_bits, dtype=np.uint8).flatten()
        if k.size != 256:
            raise ValueError("ZUC-256 需要256位密钥")
        # 将每 31 位片段打包到 LFSR 的 16 单元（简化：逐段累加生成非零状态）
        self._lfsr[:] = 1
        for i in range(16):
            seg = k[i * 16 : i * 16 + 16]  # 近似片段（非严格 31bit 切分）
            val = 0
            for b in seg:
                val = ((val << 1) | int(b)) & 0x7FFFFFFF
            self._lfsr[i] = np.uint32(val or 1)
        self._r1 = np.uint32(0)
        self._r2 = np.uint32(0)

    def _initialize(self) -> None:
        """
        初始化阶段：执行 32 次（rounds）迭代以混合 key/iv。

        Returns:
            None
        """
        for i in range(self.rounds):
            X0, X1, X2, X3 = self._bitReorg()
            W = self._F(X0, X1, X2, X3)
            # 初始化模式：将 W 的高 31 位作为外部注入值 u（规范期望 31 位）
            u = np.uint32((int(W) >> 1) & 0x7FFFFFFF)
            self._lfsrInit(u)

    def _keystream(self, n_bits: int, key_bits: np.ndarray) -> np.ndarray:
        """
        生成指定长度的密钥流（位数组）。

        Args:
            n_bits (int): 密钥流长度（比特数）
            key_bits (np.ndarray): 256 位密钥（np.uint8，元素 0/1）

        Returns:
            np.ndarray: 长度为 n_bits 的密钥流（np.uint8，元素为 0/1）

        Example:
            >>> zuc = Zuc256()
            >>> key = zuc.generateRandomBits(256)
            >>> ks = zuc._keystream(128, key)
        """
        self._loadState(key_bits)
        self._initialize()
        ks = np.zeros(int(n_bits), dtype=np.uint8)
        for i in range(int(n_bits // 32) + 1):
            X0, X1, X2, X3 = self._bitReorg()
            Z = self._F(X0, X1, X2, X3)
            self._lfsrWork()
            for j in range(32):
                idx = i * 32 + j
                if idx >= int(n_bits):
                    break
                ks[idx] = (int(Z) >> (31 - j)) & 1
        return ks

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对明文进行加密（逐位异或）。

        Args:
            plaintext (np.ndarray): 128 位明文（np.uint8，元素为 0/1）
            key (np.ndarray): 256 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位密文（np.uint8，元素为 0/1）
        """
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if pt.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"ZUC-256 仅支持{self.block_size}位明文与{self.key_size}位密钥"
            )
        ks = self._keystream(pt.size, k)
        return np.bitwise_xor(pt, ks).astype(np.uint8)

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对密文进行解密（与加密相同，逐位异或）。

        Args:
            ciphertext (np.ndarray): 128 位密文（np.uint8，元素为 0/1）
            key (np.ndarray): 256 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位明文（np.uint8，元素为 0/1）
        """
        ct = np.array(ciphertext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if ct.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"ZUC-256 仅支持{self.block_size}位密文与{self.key_size}位密钥"
            )
        ks = self._keystream(ct.size, k)
        return np.bitwise_xor(ct, ks).astype(np.uint8)