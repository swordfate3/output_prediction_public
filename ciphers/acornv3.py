import numpy as np

# [ADD] 新增: 兼容直接运行与包运行的导入方式，确保 AcornV3 与项目一致
try:
    from .base_cipher import BaseCipher  # type: ignore
except Exception:  # [ADD] 回退导入，支持在无包上下文下直接运行文件
    from ciphers.base_cipher import BaseCipher  # type: ignore


class AcornV3(BaseCipher):
    """
    AcornV3 流密码的核心实现（按 ACORN v3 规范的位级更新与输出公式）。

    说明：
    - 输入/输出为位数组（np.uint8，元素为 0/1），与项目管线一致。
    - 明文长度固定为 128 比特（`block_size=128`）以配合现有数据生成/训练流程；密钥与 IV 均为 128 比特。
    - 内部状态为 293 位寄存器，使用 ACORN v3 中的布尔函数 `maj` 与 `ch` 生成反馈位；密钥流位使用标准抽头组合。
    - 为工程可用性与简洁，当前实现专注于“密钥流 + 异或加/解密”，未包含 AEAD 认证标签生成与关联数据处理。
      若需要完整 AEAD，可在此基础上扩展控制位与消息位的馈入策略。
    """

    def __init__(self, rounds: int = 1792, iv: np.ndarray | None = None):
        """
        初始化 AcornV3 密码实例（简化版）。

        详细描述：设置固定的块与密钥长度为 128，比特数组接口；支持传入 128 位 IV，
        若未提供则使用统一的随机源生成。默认预热步数为 1792（工程建议值）。

        Args:
            rounds (int): 预热步数（工程默认 1792），用于状态充分混合
            iv (np.ndarray | None): 128 比特 IV；None 时使用统一随机源生成

        Returns:
            None

        Raises:
            ValueError: 当 IV 长度不是 128 比特时抛出异常

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(cipher.key_size)
            >>> pt = cipher.generateRandomBits(cipher.block_size)
            >>> ct = cipher.encrypt(pt, key)
        """
        # [ADD] 统一接口：block_size=128, key_size=128
        super().__init__(block_size=128, key_size=128)
        self.rounds = int(rounds)
        # [ADD] IV 统一为 128 位
        if iv is None:
            iv_bits = self.generateRandomBits(128)
        else:
            iv_bits = np.array(iv, dtype=np.uint8).flatten()
            if iv_bits.size != 128:
                raise ValueError("AcornV3 的 IV 需为128比特")
        self._iv = iv_bits

        # [ADD] 运行态寄存器：293 位状态，符合 ACORN v3 的总长度
        self._state = np.zeros(293, dtype=np.uint8)
        # [ADD] 规范抽头参数化：统一在此处定义输出位与反馈位的抽头索引
        # 说明：索引为 0 基，覆盖 z 位的线性与非线性部件，以及反馈位的线性与非线性部件。
        # 这些抽头对应 ACORN v3 的公开布尔结构描述，便于集中维护与校准。
        self._taps_z = (12, 154, 235, 0)
        self._taps_z_and = (45, 63)
        self._taps_fb_lin = (0, 107, 196)
        self._taps_fb_maj = (244, 23, 160)
        self._taps_fb_ch = (230, 111, 66)

    def _loadState(self, key_bits: np.ndarray) -> None:
        """
        加载初始状态到内部 293 位寄存器（工程化加载）。

        详细描述：
        - 将 128 位密钥加载到状态低位区域 [0..127]
        - 将 128 位 IV 加载到状态中间区域 [128..255]
        - 将剩余区域置 1 提供初始扰动（工程近似；完整规范可按 6 条 LFSR + 4 位寄存器细分装载）

        Args:
            key_bits (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）

        Returns:
            None

        Raises:
            ValueError: 当密钥长度不是 128 位时抛出异常

        Example:
            >>> cipher = AcornV3()
            >>> key = np.zeros(128, dtype=np.uint8)
            >>> cipher._loadState(key)
        """
        k = np.array(key_bits, dtype=np.uint8).flatten()
        if k.size != 128:
            raise ValueError("AcornV3 需要128位密钥")
        self._state[:] = 0
        self._state[:128] = k
        self._state[128:256] = self._iv
        # [ADD] 工程近似：将末尾区域置 1，提供初始扰动（非标准 ACORN 的装载方式）
        self._state[256:] = 1

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
            >>> AcornV3._shiftLeft(np.array([1,0,1], dtype=np.uint8), 1)
            array([0, 1, 1], dtype=uint8)
        """
        out = np.empty_like(arr)
        out[:-1] = arr[1:]
        out[-1] = int(new_bit) & 1
        return out

    @staticmethod
    def _maj(a: int, b: int, c: int) -> int:
        """
        多数函数 maj（取三者的“多数”），ACORN v3 指定用于反馈位的非线性组合。

        Args:
            a (int): 位 a（0 或 1）
            b (int): 位 b（0 或 1）
            c (int): 位 c（0 或 1）

        Returns:
            int: 结果位（0 或 1）

        Example:
            >>> AcornV3._maj(1, 0, 1)
            1
        """
        a &= 1
        b &= 1
        c &= 1
        return ((a & b) ^ (a & c) ^ (b & c)) & 1

    @staticmethod
    def _ch(a: int, b: int, c: int) -> int:
        """
        选择函数 ch（类似 SHA-2 的 choose），ACORN v3 指定用于反馈位的非线性组合。

        Args:
            a (int): 位 a（0 或 1）
            b (int): 位 b（0 或 1）
            c (int): 位 c（0 或 1）

        Returns:
            int: 结果位（0 或 1）

        Example:
            >>> AcornV3._ch(1, 0, 1)
            0
        """
        a &= 1
        b &= 1
        c &= 1
        return ((a & b) ^ ((a ^ 1) & c)) & 1

    def _feedbackBit(self) -> int:
        """
        计算 ACORN v3 的反馈比特（不含消息/控制位，工程模式）。

        详细描述：使用 `maj` 与 `ch` 组合若干抽头位，形成非线性反馈，
        该结构来源于 ACORN v3 规范（AEAD 中还会与消息位与控制位异或，这里省略）。

        Returns:
            int: 反馈比特（0 或 1）

        Example:
            >>> cipher = AcornV3()
            >>> bit = cipher._feedbackBit()
        """
        s = self._state
        # [ADD] 依据公开资料的典型反馈组合（工程模式，省略消息位与控制位）：
        # fb = S[0] ^ S[107] ^ S[196]
        # ^ maj(S[244], S[23], S[160])
        # ^ ch(S[230], S[111], S[66])
        a = int(s[244])
        b = int(s[23])
        c = int(s[160])
        d = int(s[230])
        e = int(s[111])
        f = int(s[66])
        ret = int(s[0]) ^ int(s[107]) ^ int(s[196]) ^ self._maj(a, b, c)
        ret ^= self._ch(d, e, f)
        return ret & 1

    def _warmup(self, mix_rounds: int) -> None:
        """
        预热阶段：固定执行 `mix_rounds` 次（工程默认 1792），不输出密钥流。

        Args:
            mix_rounds (int): 预热步数

        Returns:
            None

        Example:
            >>> cipher = AcornV3()
            >>> cipher._warmup(1024)
        """
        steps = int(mix_rounds)
        for _ in range(steps):
            fb = self._feedbackBit()
            # [ADD] 预热阶段：直接注入反馈位（工程模式，不含消息与控制位）
            self._state = self._shiftLeft(self._state, fb)

    def _calcZ(self) -> int:
        """
        计算密钥流输出位 z（ACORN v3 的标准抽头）。

        详细描述：使用公开抽头组合生成当前步的密钥流位，不改变内部状态。

        Returns:
            int: 密钥流位（0 或 1）

        Example:
            >>> cipher = AcornV3()
            >>> _ = cipher._loadState(cipher.generateRandomBits(128))
            >>> z = cipher._calcZ()
        """
        s = self._state
        a = int(s[self._taps_z[0]])
        b = int(s[self._taps_z[1]])
        c = int(s[self._taps_z[2]])
        d = int(s[self._taps_z[3]])
        e = int(s[self._taps_z_and[0]])
        f = int(s[self._taps_z_and[1]])
        z = (a ^ b ^ c ^ d ^ (e & f)) & 1
        return z

    def _calcFeedback(self, m_bit: int, ci_bit: int) -> int:
        """
        计算 AEAD 模式下的反馈比特（含消息位与控制位）。

        详细描述：依据 ACORN v3 的布尔结构，用 `maj/ch` 非线性组合若干抽头，
        并与消息位 `m_bit` 与控制位 `ci_bit` 进行异或，得到反馈位。

        Args:
            m_bit (int): 消息位（0 或 1），AD/明文/密文在不同阶段的位值
            ci_bit (int): 控制位（0 或 1），用于区分各阶段（初始化/AD/数据/最终化）

        Returns:
            int: 反馈位（0 或 1）

        Example:
            >>> cipher = AcornV3()
            >>> fb = cipher._calcFeedback(0, 1)
        """
        s = self._state
        m = int(m_bit) & 1
        ci = int(ci_bit) & 1
        lin = int(s[self._taps_fb_lin[0]]) ^ int(s[self._taps_fb_lin[1]])
        lin ^= int(s[self._taps_fb_lin[2]])
        a = int(s[self._taps_fb_maj[0]])
        b = int(s[self._taps_fb_maj[1]])
        c = int(s[self._taps_fb_maj[2]])
        d = int(s[self._taps_fb_ch[0]])
        e = int(s[self._taps_fb_ch[1]])
        f = int(s[self._taps_fb_ch[2]])
        return (lin ^ self._maj(a, b, c) ^ self._ch(d, e, f) ^ m ^ ci) & 1

    def _clockAead(self, m_bit: int, ci_bit: int) -> int:
        """
        AEAD 的一次时钟步：计算 z、计算反馈并更新状态。

        详细描述：先读取密钥流位 `z`（不修改状态），随后计算反馈位并注入（左移+最低位），
        用于初始化、吸收关联数据、处理数据与最终化生成标签等阶段。

        Args:
            m_bit (int): 消息位（0 或 1），当前阶段输入的位
            ci_bit (int): 控制位（0 或 1），标识阶段：1=初始化/AD/最终化，0=数据处理

        Returns:
            int: 密钥流位 z（0 或 1）

        Example:
            >>> cipher = AcornV3()
            >>> z = cipher._clockAead(0, 1)
        """
        z = self._calcZ()
        fb = self._calcFeedback(m_bit, ci_bit)
        self._state = self._shiftLeft(self._state, fb)
        return z

    def initializeAead(self, key_bits: np.ndarray, iv_bits: np.ndarray) -> None:
        """
        初始化 AEAD 状态：加载密钥与 IV 并执行预热混合。

        详细描述：
        按位吸收密钥与 IV（ci=1, m=bit），随后使用 `ci=1, m=0` 执行
        `rounds` 次预热混合，使初始状态充分扩散。

        Args:
            key_bits (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）
            iv_bits (np.ndarray): 128 位 IV（np.uint8，元素为 0/1）

        Returns:
            None

        Raises:
            ValueError: 当输入长度不匹配时抛出异常

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(128)
            >>> iv = cipher.generateRandomBits(128)
            >>> cipher.initializeAead(key, iv)
        """
        iv_bits = np.array(iv_bits, dtype=np.uint8).flatten()
        if iv_bits.size != 128:
            raise ValueError("AcornV3 的 IV 需为128比特")
        self._iv = iv_bits
        # [MOD] 规范化吸收：按位注入 key/iv（ci=1），而非直接写入状态
        # 清空状态
        self._state[:] = 0
        # 注入密钥位（ci=1, m=key_i）
        k_arr = np.array(key_bits, dtype=np.uint8).flatten()
        if k_arr.size != 128:
            raise ValueError("AcornV3 的密钥需为128比特")
        for bit in k_arr:
            _ = self._clockAead(int(bit) & 1, 1)
        # 注入 IV 位（ci=1, m=iv_i）
        for bit in iv_bits:
            _ = self._clockAead(int(bit) & 1, 1)
        # 预热混合：ci=1, m=0
        for _ in range(int(self.rounds)):
            _ = self._clockAead(0, 1)

    def absorbAssociatedData(self, ad_bits: np.ndarray) -> None:
        """
        吸收关联数据（AD），用于 AEAD 的鉴别部分。

        详细描述：遍历关联数据位序列，对每一位执行 `ci=1, m=ad_i` 的时钟步，
        不产生密文，仅更新内部状态以影响最终标签。

        Args:
            ad_bits (np.ndarray): 关联数据位数组（np.uint8，元素为 0/1）

        Returns:
            None

        Example:
            >>> cipher = AcornV3()
            >>> ad = cipher.generateRandomBits(64)
            >>> cipher.absorbAssociatedData(ad)
        """
        ad_arr = np.array(ad_bits, dtype=np.uint8).flatten()
        for bit in ad_arr:
            _ = self._clockAead(int(bit) & 1, 1)

    def encryptAead(
        self,
        plaintext_bits: np.ndarray,
        key_bits: np.ndarray,
        iv_bits: np.ndarray,
        associated_data_bits: np.ndarray | None = None,
        tag_bits_length: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ACORN v3 AEAD 加密：生成密文与认证标签。

        详细描述：
        1) 初始化：加载密钥与 IV 并预热混合（ci=1, m=0）；
        2) 吸收关联数据：ci=1, m=ad_i；
        3) 加密明文：逐位 z = calcZ；c_i = p_i ^ z；更新状态：ci=0, m=p_i；
        4) 最终化：生成标签位：重复 `tag_bits_length` 次 z 输出，每步更新状态：ci=1, m=0。

        Args:
            plaintext_bits (np.ndarray): 明文位数组（np.uint8，元素为 0/1）
            key_bits (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）
            iv_bits (np.ndarray): 128 位 IV（np.uint8，元素为 0/1）
            associated_data_bits (np.ndarray | None): 关联数据位数组；None 表示无 AD
            tag_bits_length (int): 标签长度（位数），默认 128

        Returns:
            tuple[np.ndarray, np.ndarray]: (ciphertext_bits, tag_bits)

        Raises:
            ValueError: 当输入长度不匹配时抛出异常

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(128)
            >>> iv = cipher.generateRandomBits(128)
            >>> pt = cipher.generateRandomBits(128)
            >>> ct, tag = cipher.encryptAead(pt, key, iv)
        """
        pt = np.array(plaintext_bits, dtype=np.uint8).flatten()
        if pt.size <= 0:
            raise ValueError("明文长度必须为正")
        if np.array(key_bits).flatten().size != 128:
            raise ValueError("密钥长度必须为 128 位")
        if np.array(iv_bits).flatten().size != 128:
            raise ValueError("IV 长度必须为 128 位")

        # 初始化与吸收 AD
        self.initializeAead(key_bits, iv_bits)
        if associated_data_bits is not None:
            self.absorbAssociatedData(associated_data_bits)

        # 加密阶段
        ct = np.zeros(pt.size, dtype=np.uint8)
        for i, p in enumerate(pt):
            z = self._calcZ()
            c = (int(p) ^ z) & 1
            ct[i] = c
            # ci=0, m=plaintext_bit
            _ = self._clockAead(int(p) & 1, 0)

        # 最终化生成标签
        tag_len = int(tag_bits_length)
        tag = np.zeros(tag_len, dtype=np.uint8)
        for i in range(tag_len):
            z = self._calcZ()
            tag[i] = z & 1
            # ci=1, m=0（最终化步）
            _ = self._clockAead(0, 1)

        return ct, tag

    def decryptAead(
        self,
        ciphertext_bits: np.ndarray,
        key_bits: np.ndarray,
        iv_bits: np.ndarray,
        associated_data_bits: np.ndarray | None,
        tag_bits: np.ndarray,
    ) -> np.ndarray:
        """
        ACORN v3 AEAD 解密：恢复明文并验证认证标签。

        详细描述：
        1) 初始化：加载密钥与 IV 并预热混合；
        2) 吸收关联数据；
        3) 解密：逐位 z = calcZ；p_i = c_i ^ z；更新状态：ci=0, m=p_i；
        4) 最终化：生成标签并与提供的 `tag_bits` 比较，不匹配则抛出异常。

        Args:
            ciphertext_bits (np.ndarray): 密文位数组（np.uint8，元素为 0/1）
            key_bits (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）
            iv_bits (np.ndarray): 128 位 IV（np.uint8，元素为 0/1）
            associated_data_bits (np.ndarray | None): 关联数据位数组；None 表示无 AD
            tag_bits (np.ndarray): 认证标签位数组（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 明文位数组（np.uint8，元素为 0/1）

        Raises:
            ValueError: 当标签不匹配或输入非法时抛出异常

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(128)
            >>> iv = cipher.generateRandomBits(128)
            >>> pt = cipher.generateRandomBits(128)
            >>> ct, tag = cipher.encryptAead(pt, key, iv)
            >>> pt2 = cipher.decryptAead(ct, key, iv, None, tag)
            >>> np.array_equal(pt, pt2)
            True
        """
        ct = np.array(ciphertext_bits, dtype=np.uint8).flatten()
        tag_arr = np.array(tag_bits, dtype=np.uint8).flatten()
        if ct.size <= 0:
            raise ValueError("密文长度必须为正")
        if np.array(key_bits).flatten().size != 128:
            raise ValueError("密钥长度必须为 128 位")
        if np.array(iv_bits).flatten().size != 128:
            raise ValueError("IV 长度必须为 128 位")
        if tag_arr.size <= 0:
            raise ValueError("标签长度必须为正")

        # 初始化与吸收 AD
        self.initializeAead(key_bits, iv_bits)
        if associated_data_bits is not None:
            self.absorbAssociatedData(associated_data_bits)

        # 解密阶段
        pt = np.zeros(ct.size, dtype=np.uint8)
        for i, c in enumerate(ct):
            z = self._calcZ()
            p = (int(c) ^ z) & 1
            pt[i] = p
            # ci=0, m=plaintext_bit（保证与加密侧状态一致）
            _ = self._clockAead(int(p) & 1, 0)

        # 最终化生成标签并校验
        gen_tag = np.zeros(tag_arr.size, dtype=np.uint8)
        for i in range(tag_arr.size):
            z = self._calcZ()
            gen_tag[i] = z & 1
            _ = self._clockAead(0, 1)

        if not np.array_equal(gen_tag, tag_arr):
            raise ValueError("AEAD 标签验证失败：不匹配")

        return pt

    def _keystream(self, n_bits: int, key_bits: np.ndarray) -> np.ndarray:
        """
        生成指定长度的密钥流（按 ACORN v3 的标准抽头计算 z）。

        详细描述：统一采用 AEAD 的初始化与时序：
        - 使用 `initializeAead(key_bits, self._iv)` 执行预热（ci=1, m=0）以加载 key/iv
        - 生成密钥流阶段：每步先读取 z，再执行一次 `ci=0, m=0` 的工作步更新状态
        密钥流位使用公开抽头组合：z = S[12] ^ S[154] ^ S[235] ^ S[0] ^ (S[45] & S[63])。

        Args:
            n_bits (int): 密钥流长度（比特数）
            key_bits (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 长度为 n_bits 的密钥流（np.uint8，元素为 0/1）

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(128)
            >>> ks = cipher._keystream(128, key)
        """
        # [MOD] 统一到 AEAD 初始化流程，确保与 AEAD 路径一致的装载与混合
        self.initializeAead(key_bits, self._iv)
        ks = np.zeros(int(n_bits), dtype=np.uint8)
        for i in range(int(n_bits)):
            # [ADD] 标准密钥流输出位（公开抽头组合）
            z = self._calcZ()
            ks[i] = z
            # [MOD] 工作阶段更新：使用 AEAD 的时序（ci=0, m=0）
            _ = self._clockAead(0, 0)
        return ks
        return ks

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对明文进行加密（逐位异或）。

        详细描述：生成与明文等长的密钥流，然后逐位异或得到密文。

        Args:
            plaintext (np.ndarray): 128 位明文（np.uint8，元素为 0/1）
            key (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位密文（np.uint8，元素为 0/1）

        Raises:
            ValueError: 当输入长度不匹配时抛出

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(cipher.key_size)
            >>> pt = cipher.generateRandomBits(cipher.block_size)
            >>> ct = cipher.encrypt(pt, key)
        """
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if pt.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"AcornV3 仅支持{self.block_size}位明文与{self.key_size}位密钥"
            )
        ks = self._keystream(pt.size, k)
        return np.bitwise_xor(pt, ks).astype(np.uint8)

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        使用密钥流对密文进行解密（与加密相同，逐位异或）。

        详细描述：生成与密文等长的密钥流，然后逐位异或得到明文。

        Args:
            ciphertext (np.ndarray): 128 位密文（np.uint8，元素为 0/1）
            key (np.ndarray): 128 位密钥（np.uint8，元素为 0/1）

        Returns:
            np.ndarray: 128 位明文（np.uint8，元素为 0/1）

        Raises:
            ValueError: 当输入长度不匹配时抛出

        Example:
            >>> cipher = AcornV3()
            >>> key = cipher.generateRandomBits(cipher.key_size)
            >>> ct = cipher.generateRandomBits(cipher.block_size)
            >>> pt = cipher.decrypt(ct, key)
        """
        ct = np.array(ciphertext, dtype=np.uint8).flatten()
        k = np.array(key, dtype=np.uint8).flatten()
        if ct.size != self.block_size or k.size != self.key_size:
            raise ValueError(
                f"AcornV3 仅支持{self.block_size}位密文与{self.key_size}位密钥"
            )
        ks = self._keystream(ct.size, k)
        return np.bitwise_xor(ct, ks).astype(np.uint8)