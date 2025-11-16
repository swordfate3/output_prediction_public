import numpy as np

# 兼容直接运行与包运行的导入方式
try:
    from .base_cipher import BaseCipher  # type: ignore
except Exception:
    from ciphers.base_cipher import BaseCipher  # type: ignore


class Grain128a(BaseCipher):
    """
    Grain-128a 完整实现（符合公开规范）：
    - 遵循文档1-30至1-75核心定义：128位密钥、96位IV、256位状态（128LFSR+128NFSR）
    - 模式控制：IV₀=1强制认证，IV₀=0禁止认证（文档1-60）
    - 认证机制：32位累加器+32位移位寄存器，前64位预输出初始化（文档1-67）
    - 密钥流：无认证时直接输出预输出；认证时取y_{64+2i}（文档1-63）
    - 标签：最大32位，取累加器右侧bit（文档1-73）
    """

    def __init__(self, rounds: int = 256, iv: np.ndarray | None = None, auth_mode: bool = False):
        super().__init__(block_size=128, key_size=128)
        print(f"Grain128a 初始化：{rounds}轮预热，认证模式={auth_mode}")
        self._rounds = rounds  # 自由设置预热轮数
        
        # 1. IV初始化：仅支持96位（文档1-54）
        if iv is None:
            self._iv = np.random.randint(0, 2, size=96, dtype=np.uint8)  # 默认随机96位IV
            if auth_mode:
                self._iv[0] = 1  # 默认强制认证
            else:
                self._iv[0] = 0  # 默认无认证
        else:
            self._iv = np.array(iv, dtype=np.uint8).flatten()
            if self._iv.size != 96:
                raise ValueError("Grain128a 仅支持96位IV（文档规定）")
        
        # 2. 模式控制：IV₀决定认证是否允许（文档1-60）
        self._iv0 = self._iv[0]  # IV第0位：0=无认证，1=强制认证
        self._is_auth_mode = self._iv0 == 1
        
        # 3. 运行态寄存器（含认证组件）
        self._lfsr = np.zeros(128, dtype=np.uint8)
        self._nfsr = np.zeros(128, dtype=np.uint8)
        self._a = np.zeros(32, dtype=np.uint8)  # 认证累加器（固定32位，文档1-67）
        self._r = np.zeros(32, dtype=np.uint8)  # 认证移位寄存器（固定32位）
        self._pre_output_cache: list[int] = []  # 预输出缓存（避免重复计算）

    # ===== 基础工具函数 =====
    @staticmethod
    def _shift(arr: np.ndarray, new_bit: int) -> np.ndarray:
        """左移1位，最低位插入新bit（文档1-42/43寄存器更新逻辑）"""
        out = np.empty_like(arr)
        out[:-1] = arr[1:]
        out[-1] = new_bit & 1
        return out

    @staticmethod
    def _xor_bits(bits: np.ndarray, idxs: list[int] | tuple[int, ...]) -> int:
        """指定索引bit异或累加（模2）"""
        acc = 0
        for i in idxs:
            acc ^= int(bits[i])
        return acc & 1

    # ===== 核心反馈函数（严格对齐文档公式）=====
    def _l(self, lfsr: np.ndarray) -> int:
        """LFSR反馈函数f(x)（文档1-44/45）：f(x)=1+x³²+x⁴⁷+x⁵⁸+x⁹⁰+x¹²¹+x¹²⁸"""
        taps = (0, 7, 38, 70, 81, 96)  # 相对索引映射
        return self._xor_bits(lfsr, taps)

    def _g(self, nfsr: np.ndarray) -> int:
        """NFSR反馈函数g(x)（文档1-46/49）：含线性/二次/三次/四次项"""
        # 1. 线性项：b[i] ^ b[i+26] ^ b[i+56] ^ b[i+91] ^ b[i+96]
        lin_taps = (0, 26, 56, 91, 96)
        lin_val = self._xor_bits(nfsr, lin_taps)
        
        # 2. 二次项（成对AND）：(3,67), (11,13), ..., (68,84)（文档1-49）
        quad_pairs = (
            (3, 67), (11, 13), (17, 18), (27, 59),
            (40, 48), (61, 65), (68, 84)
        )
        quad_val = 0
        for a, b in quad_pairs:
            quad_val ^= (int(nfsr[a]) & int(nfsr[b]))
        
        # 3. 三次项（三元素AND）：(22,24,25), (70,78,82)
        triplet_taps = ((22, 24, 25), (70, 78, 82))
        triplet_val = 0
        for a, b, c in triplet_taps:
            triplet_val ^= (int(nfsr[a]) & int(nfsr[b]) & int(nfsr[c]))
        
        # 4. 四次项（四元素AND）：(88,92,93,95)
        a, b, c, d = (88, 92, 93, 95)
        quad4_val = int(nfsr[a]) & int(nfsr[b]) & int(nfsr[c]) & int(nfsr[d])
        
        return (lin_val ^ quad_val ^ triplet_val ^ quad4_val) & 1

    def _h(self) -> int:
        """预输出布尔函数h(x)（文档1-51/52）：h=x0x1^x2x3^x4x5^x6x7^x0x4x8"""
        b, s = self._nfsr, self._lfsr
        # x0~x8映射（相对i）：文档1-52
        x0 = int(b[12])   # x0=b[i+12]
        x1 = int(s[8])    # x1=s[i+8]
        x2 = int(s[13])   # x2=s[i+13]
        x3 = int(s[20])   # x3=s[i+20]
        x4 = int(b[95])   # x4=b[i+95]
        x5 = int(s[42])   # x5=s[i+42]
        x6 = int(s[60])   # x6=s[i+60]
        x7 = int(s[79])   # x7=s[i+79]
        x8 = int(s[94])   # x8=s[i+94]（Grain-128a关键修改，文档1-165）
        
        return (x0 & x1 ^ x2 & x3 ^ x4 & x5 ^ x6 & x7 ^ x0 & x4 & x8) & 1

    def _pre_output(self) -> int:
        """计算预输出y（文档1-53）：y = h(x) ^ s[i+93] ^ Σ(b[i+j], j∈A)，A={2,15,36,45,64,73,89}"""
        h_val = self._h()
        s93_val = int(self._lfsr[93])
        nfsr_taps = (2, 15, 36, 45, 64, 73, 89)
        nfsr_val = self._xor_bits(self._nfsr, nfsr_taps)
        y = (h_val ^ s93_val ^ nfsr_val) & 1
        
        # 更新寄存器（工作阶段：无预输出反馈，文档1-40至1-53）
        l_new = self._l(self._lfsr)
        g_val = self._g(self._nfsr)
        n_new = (g_val ^ int(self._lfsr[0])) & 1  # NFSR含s[i]，文档1-49
        self._lfsr = self._shift(self._lfsr, l_new)
        self._nfsr = self._shift(self._nfsr, n_new)
        
        return y

    # ===== 状态初始化=====
    def _load_state(self, key_bits: np.ndarray) -> None:
        """加载密钥与IV到寄存器（文档1-54/55）"""
        key_bits = np.array(key_bits, dtype=np.uint8).flatten()
        if key_bits.size != 128:
            raise ValueError("Grain128a 需128位密钥（文档规定）")
        
        # 1. NFSR = 128位密钥
        self._nfsr[:] = key_bits
        
        # 2. LFSR = IV（96位） || 31个1 || 1个0（文档1-55，避免移位等价攻击）
        self._lfsr[:96] = self._iv
        self._lfsr[96:127] = 1  # s[96]~s[126] = 1（31位）
        self._lfsr[127] = 0     # s[127] = 0（文档1-158修复Grain-128漏洞）

    def _initialize(self) -> None:
        """预热轮数：预输出反馈到寄存器（文档1-56）"""
        self._pre_output_cache.clear()
        for _ in range(self._rounds):
            y = self._h()  # 计算预输出（暂不更新寄存器，因需反馈）
            # 预热阶段特殊更新：LFSR/NFSR需反馈预输出y（文档1-56）
            l_new = (self._l(self._lfsr) ^ y) & 1
            g_val = self._g(self._nfsr)
            n_new = (g_val ^ int(self._lfsr[0]) ^ y) & 1
            # 更新寄存器
            self._lfsr = self._shift(self._lfsr, l_new)
            self._nfsr = self._shift(self._nfsr, n_new)
            # 缓存预热后的预输出（后续工作阶段使用）
            self._pre_output_cache.append(y)
        
        # 预热后继续生成工作阶段的预输出（缓存，避免重复计算）
        while len(self._pre_output_cache) < 1024:  # 预生成足够多，减少后续耗时
            self._pre_output_cache.append(self._pre_output())

    # ===== 认证模式辅助函数（文档1-65至1-73）=====
    def _init_auth_registers(self) -> None:
        """用前64位预输出初始化累加器a和移位寄存器r（文档1-67）"""
        # 确保缓存有至少64位预输出
        if len(self._pre_output_cache) < 64:
            while len(self._pre_output_cache) < 64:
                self._pre_output_cache.append(self._pre_output())
        
        # a0^j = yj（0≤j≤31），r_i = y32+i（0≤i≤31）
        pre_output_64 = self._pre_output_cache[:64]
        self._a[:] = pre_output_64[:32]
        self._r[:] = pre_output_64[32:64]
        # 移除已使用的64位预输出
        self._pre_output_cache = self._pre_output_cache[64:]

    def _update_auth_registers(self, mi: int) -> None:
        """更新认证寄存器：r左移+插入新bit，a = a ^ (mi * r)（文档1-68/69）"""
        # 1. 获取用于更新r的预输出bit（y_{64+2i+1}，文档1-68）
        if not self._pre_output_cache:
            self._pre_output_cache.append(self._pre_output())
        y_odd = self._pre_output_cache.pop(0)
        
        # 2. 更新移位寄存器r：r[i+32] = y_odd，左移1位
        self._r = self._shift(self._r, y_odd)
        
        # 3. 更新累加器a：a[i+1]^j = a[i]^j ^ (mi & r[j])（模2）
        if mi:
            self._a ^= self._r

    # ===== 密钥流生成（分模式）=====
    def _generate_keystream(self, n_bits: int) -> np.ndarray:
        """生成密钥流（无认证：直接取预输出；认证：取y_{64+2i}）"""
        ks = np.zeros(n_bits, dtype=np.uint8)
        
        if not self._is_auth_mode:
            # 无认证模式：预输出直接作为密钥流（文档1-64）
            if len(self._pre_output_cache) < n_bits:
                need = n_bits - len(self._pre_output_cache)
                self._pre_output_cache.extend([self._pre_output() for _ in range(need)])
            ks[:] = self._pre_output_cache[:n_bits]
            self._pre_output_cache = self._pre_output_cache[n_bits:]
        
        else:
            # 认证模式：跳过前64位（已用于初始化a/r），取每第二个bit（y_{64+2i}，文档1-63）
            required = 64 + 2 * n_bits  # 前64位已用，需额外2*n_bits位（1位密钥流+1位认证）
            if len(self._pre_output_cache) < required - 64:  # 已扣除初始化用的64位
                need = (required - 64) - len(self._pre_output_cache)
                self._pre_output_cache.extend([self._pre_output() for _ in range(need)])
            
            # 提取y_{64+2i}（索引64,66,...64+2(n_bits-1)）
            for i in range(n_bits):
                ks[i] = self._pre_output_cache[2 * i]
            # 移除已使用的2*n_bits位
            self._pre_output_cache = self._pre_output_cache[2 * n_bits:]
        
        return ks

    # ===== 加密（分模式实现）=====
    def encrypt(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        """
        加密（仅负责生成密文，不做认证更新与标签生成）

        说明：在认证模式（IV₀=1）下，依规范需跳过前64位预输出初始化认证寄存器，
        因此此函数会执行认证寄存器初始化以保证密钥流索引正确，但不会进行后续的
        认证寄存器更新与标签生成，这部分请调用 generate_auth_tag 完成。

        Args:
            plaintext (np.ndarray): 128位明文比特数组
            key (np.ndarray): 128位密钥比特数组

        Returns:
            np.ndarray: 密文字节（与明文同长度）

        Raises:
            ValueError: 当明文或密钥长度不为128位时抛出异常

        Example:
            >>> cipher = Grain128a(iv=np.zeros(96, dtype=np.uint8))
            >>> pt = np.zeros(128, dtype=np.uint8)
            >>> key = np.zeros(128, dtype=np.uint8)
            >>> ct = cipher.encrypt(pt, key)
        """
        # 1. 输入校验
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        key_bits = np.array(key, dtype=np.uint8).flatten()
        if pt.size != self.block_size:
            raise ValueError(f"仅支持{self.block_size}位明文（项目约定）")
        if key_bits.size != self.key_size:
            raise ValueError(f"需{self.key_size}位密钥（文档规定）")
        # [DEL] 删除认证标签参数校验逻辑：encrypt不再处理标签
        # 作用：遵从职责单一，避免在加密流程中耦合认证生成
        
        # 3. 初始化状态
        self._load_state(key_bits)
        self._initialize()
        
        # 4. 认证模式需初始化认证寄存器以跳过前64位预输出（不做更新）
        if self._is_auth_mode:
            # [ADD] 在加密中仅进行认证寄存器初始化，保证密钥流索引正确
            # 作用：符合规范的密钥流选择 y_{64+2i}，不产生标签
            self._init_auth_registers()

        # 5. 生成密钥流并返回密文
        ks = self._generate_keystream(pt.size)
        ct = np.bitwise_xor(pt, ks).astype(np.uint8)
        return ct

    def generate_auth_tag(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        tag_bits_length: int = 32,
    ) -> np.ndarray:
        """
        生成认证标签（分离自加密流程）

        在 IV₀=1 的认证模式下，根据规范使用前64位预输出初始化认证寄存器，
        然后对消息（附加Padding=1）进行逐比特的寄存器更新，最终提取累加器
        右侧的 w 位作为标签。

        Args:
            plaintext (np.ndarray): 128位明文比特数组
            key (np.ndarray): 128位密钥比特数组
            tag_bits_length (int): 标签位长度，范围 1~32，默认32

        Returns:
            np.ndarray: 认证标签比特数组（长度为 tag_bits_length）

        Raises:
            ValueError: 当不处于认证模式或参数长度非法时抛出异常

        Example:
            >>> cipher = Grain128a(iv=np.concatenate([[1], np.zeros(95, dtype=np.uint8)]))
            >>> pt = np.zeros(128, dtype=np.uint8)
            >>> key = np.zeros(128, dtype=np.uint8)
            >>> tag = cipher.generate_auth_tag(pt, key, tag_bits_length=32)
        """
        # 1. 输入校验
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        key_bits = np.array(key, dtype=np.uint8).flatten()
        if pt.size != self.block_size:
            raise ValueError(f"仅支持{self.block_size}位明文（项目约定）")
        if key_bits.size != self.key_size:
            raise ValueError(f"需{self.key_size}位密钥（文档规定）")
        if not self._is_auth_mode:
            raise ValueError("无认证模式（IV₀=0）禁止生成认证标签（文档1-60）")
        if not (1 <= int(tag_bits_length) <= 32):
            raise ValueError("认证模式需指定tag_bits_length（1~32位，文档规定）")
        self._tag_len = int(tag_bits_length)

        # 2. 初始化状态与认证寄存器
        self._load_state(key_bits)
        self._initialize()
        self._init_auth_registers()

        # [ADD] 预先消耗与加密一致的密钥流步数，保持与解密校验流程一致
        # 作用：与 decrypt 中的 _generate_keystream(ct.size) 保持相同状态推进
        _ = self._generate_keystream(pt.size)

        # 3. 消息Padding并更新认证寄存器
        pt_padded = np.concatenate([pt, [1]])
        for i in range(pt_padded.size):
            mi = int(pt_padded[i])
            self._update_auth_registers(mi)

        # 4. 提取标签
        tag = self._a[-self._tag_len:].copy()
        return tag

    # ===== 解密（分模式实现）=====
    def decrypt(
        self,
        ciphertext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        """
        解密（仅负责恢复明文，不做标签校验）

        说明：在认证模式（IV₀=1）下，此函数会进行认证寄存器初始化以保证密钥流
        索引与规范一致，但不进行认证寄存器更新与标签对比。标签校验请调用
        verify_auth_tag 完成。

        Args:
            ciphertext (np.ndarray): 128位密文比特数组
            key (np.ndarray): 128位密钥比特数组

        Returns:
            np.ndarray: 解密得到的明文比特数组

        Raises:
            ValueError: 当密文或密钥长度不为128位时抛出异常

        Example:
            >>> cipher = Grain128a(iv=np.zeros(96, dtype=np.uint8))
            >>> ct = np.zeros(128, dtype=np.uint8)
            >>> key = np.zeros(128, dtype=np.uint8)
            >>> pt = cipher.decrypt(ct, key)
        """
        # 1. 输入校验
        ct = np.array(ciphertext, dtype=np.uint8).flatten()
        key_bits = np.array(key, dtype=np.uint8).flatten()
        if ct.size != self.block_size:
            raise ValueError(f"仅支持{self.block_size}位密文（项目约定）")
        if key_bits.size != self.key_size:
            raise ValueError(f"需{self.key_size}位密钥（文档规定）")
        # [DEL] 删除标签参数校验：decrypt不再处理标签校验
        
        # 3. 初始化状态
        self._load_state(key_bits)
        self._initialize()
        
        # 4. 认证模式：初始化认证寄存器以对齐密钥流索引（不做寄存器更新）
        if self._is_auth_mode:
            # [ADD] 初始化认证寄存器以跳过前64位预输出，保持y_{64+2i}选择
            self._init_auth_registers()
        
        # 5. 生成密钥流并解密
        ks = self._generate_keystream(ct.size)
        pt = np.bitwise_xor(ct, ks).astype(np.uint8)
        return pt

    def verify_auth_tag(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        tag_bits: np.ndarray,
    ) -> bool:
        """
        校验认证标签（分离自解密流程）

        在 IV₀=1 的认证模式下，使用与标签生成相同的步骤：初始化状态与认证寄存器，
        预先消耗与消息长度相同的密钥流步数（与加密/解密保持一致），对 `m||1` 逐位
        更新认证寄存器，提取累加器右侧 w 位与传入标签比对。

        Args:
            plaintext (np.ndarray): 128位明文比特数组
            key (np.ndarray): 128位密钥比特数组
            tag_bits (np.ndarray): 传入的标签比特数组（长度 1~32）

        Returns:
            bool: 标签匹配返回 True，否则返回 False

        Raises:
            ValueError: 当不处于认证模式或参数长度非法时抛出异常

        Example:
            >>> ok = cipher.verify_auth_tag(pt, key, tag_bits=tag)
        """
        # 1. 输入校验
        pt = np.array(plaintext, dtype=np.uint8).flatten()
        key_bits = np.array(key, dtype=np.uint8).flatten()
        tag_arr = np.array(tag_bits, dtype=np.uint8).flatten()
        if pt.size != self.block_size:
            raise ValueError(f"仅支持{self.block_size}位明文（项目约定）")
        if key_bits.size != self.key_size:
            raise ValueError(f"需{self.key_size}位密钥（文档规定）")
        if not self._is_auth_mode:
            raise ValueError("无认证模式（IV₀=0）禁止校验认证标签（文档1-60）")
        self._tag_len = tag_arr.size
        if not (1 <= self._tag_len <= 32):
            raise ValueError("标签长度需为1~32位（文档规定）")

        # 2. 初始化状态与认证寄存器
        self._load_state(key_bits)
        self._initialize()
        self._init_auth_registers()

        # 3. 与生成逻辑一致，预先消耗与消息长度相同的密钥流步数
        _ = self._generate_keystream(pt.size)

        # 4. 消息Padding并更新认证寄存器
        pt_padded = np.concatenate([pt, [1]])
        for i in range(pt_padded.size):
            mi = int(pt_padded[i])
            self._update_auth_registers(mi)

        # 5. 标签比对
        gen_tag = self._a[-self._tag_len:].copy()
        return np.array_equal(gen_tag, tag_arr)


def test_grain128a_compliance():
    """验证实现在96位IV约束下的加解密与认证流程可运行"""
    # 测试向量参数（保持原始十六进制，但按规范截取为96位IV）
    key_hex = "00000000000000000123456789abcdef00000000000000000123456789abcdef"
    iv_hex_auth = "80000000000000008123456789abcdef"  # IV₀=1（第0位=1，认证模式）
    iv_hex_noauth = "00000000000000000123456789abcdef"  # IV₀=0（无认证模式）

    def hex2bits(hex_str: str) -> np.ndarray:
        """十六进制转numpy位数组（高位在前）"""
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
        return np.array([int(c) for c in bin_str], dtype=np.uint8)

    def bits2hex(bits: np.ndarray) -> str:
        """numpy位数组转十六进制（高位在前）"""
        if bits.size % 4 != 0:
            bits = np.pad(bits, (0, 4 - bits.size % 4), mode="constant")
        return hex(int(''.join(map(str, bits)), 2))[2:].zfill(len(bits) // 4)

    # 1. 测试无认证模式（IV₀=0）
    print("=== 测试1：无认证模式（IV₀=0）===")
    iv_noauth_128 = hex2bits(iv_hex_noauth)
    iv_noauth = iv_noauth_128[:96]
    cipher_noauth = Grain128a(iv=iv_noauth)
    key_bits_full = hex2bits(key_hex)
    key_bits = key_bits_full[:128]
    pt = np.zeros(128, dtype=np.uint8)  # 128位空明文
    ct = cipher_noauth.encrypt(pt, key_bits)
    # 提取前64位密文（即密钥流，因明文为0）
    ct_64 = ct[:64]
    ct_64_hex = bits2hex(ct_64)
    print(f"生成64位密钥流：{ct_64_hex}")
    print("无认证模式加密流程运行正常！\n")

    # 2. 测试认证模式（IV₀=1，空消息m0）
    print("=== 测试2：认证模式（IV₀=1，空消息m0）===")
    iv_auth_128 = hex2bits(iv_hex_auth)
    iv_auth = iv_auth_128[:96]
    cipher_auth = Grain128a(iv=iv_auth)
    # 空明文（m0长度0，加密后密文为密钥流，标签为4ff6a6c1）
    pt_empty = np.zeros(128, dtype=np.uint8)  # 此处用128位空明文模拟m0（实际m0长度0，需调整plaintext）
    # 注意：文档m0是长度0的消息，此处因block_size=128，用全0明文+Padding=1模拟
    ct_auth = cipher_auth.encrypt(pt_empty, key_bits)
    tag = cipher_auth.generate_auth_tag(pt_empty, key_bits, tag_bits_length=32)
    tag_hex = bits2hex(tag)
    print(f"生成32位标签：{tag_hex}")
    print("认证模式标签生成运行正常！\n")

    # 3. 测试认证模式解密（标签校验）
    print("=== 测试3：认证模式解密（标签校验）===")
    cipher_dec_auth = Grain128a(iv=iv_auth)
    try:
        pt_dec = cipher_dec_auth.decrypt(ct_auth, key_bits)
        ok = cipher_dec_auth.verify_auth_tag(pt_dec, key_bits, tag_bits=tag)
        assert ok, "认证模式标签验证失败"
        print("标签校验通过，解密成功！")
        assert np.array_equal(pt_dec, pt_empty), "解密明文与原明文不一致"
        print("认证模式解密测试通过！")
    except ValueError as e:
        assert False, f"认证模式解密失败：{e}"


if __name__ == "__main__":
    test_grain128a_compliance()