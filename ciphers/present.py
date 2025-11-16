import numpy as np

# [DEL] 删除：条件导入回退逻辑，使用标准相对导入简化
from .base_cipher import BaseCipher


class SmallPRESENT4(BaseCipher):
    """论文2.1节：16位SPN结构small PRESENT-[4]"""

    def __init__(self, rounds=4):
        super().__init__(block_size=16, key_size=80)
        # 论文表2：PRESENT原始S盒（4bit输入→4bit输出）
        self.rounds = rounds
        self.S_BOX = np.array(
            [
                0xC,
                0x5,
                0x6,
                0xB,
                0x9,
                0x0,
                0xA,
                0xD,
                0x3,
                0xE,
                0xF,
                0x8,
                0x4,
                0x7,
                0x1,
                0x2,
            ],
            dtype=np.uint8,
        )

    def _p_layer(self, state: np.ndarray) -> np.ndarray:
        """
        论文定义的置换层：位置 i 映射到 4*i mod 15（i<15），i=15 映射到 15

        采用列表构造置换索引并进行花式索引，避免使用 np.empty_like。
        """
        permutation = [((4 * i) % 15) if i < 15 else 15 for i in range(16)]
        return state[permutation]

    def _inv_p_layer(self, state: np.ndarray) -> np.ndarray:
        """
        逆置换层

        详细描述：实现 small PRESENT-[4] 的置换层的逆映射。
        原置换定义为位置 i → (4*i mod 15)（i<15），i=15 → 15。
        因此逆置换通过构造原置换的逆索引实现。

        Args:
            state (np.ndarray): 16 位状态向量（np.uint8，元素为 0/1）。

        Returns:
            np.ndarray: 经过逆置换后的 16 位状态向量。

        Example:
            >>> import numpy as np
            >>> sp = SmallPRESENT4()
            >>> s = np.arange(16, dtype=np.uint8) % 2
            >>> inv = sp._inv_p_layer(sp._p_layer(s))
            >>> np.array_equal(inv, s)
            True
        """
        # 简化：省略类型检查，直接构造逆置换
        inv_permutation = [0] * 16
        permutation = [((4 * i) % 15) if i < 15 else 15 for i in range(16)]
        for i, p in enumerate(permutation):
            inv_permutation[p] = i
        return state[inv_permutation]

    def _inv_s_box(self) -> np.ndarray:
        """
        逆S盒

        详细描述：根据 small PRESENT-[4] 的 S 盒定义，构造其逆映射表，
        用于解密时的逆替换操作。

        Returns:
            np.ndarray: 长度为16的逆S盒查找表（np.uint8）。

        Example:
            >>> sp = SmallPRESENT4()
            >>> inv = sp._inv_s_box()
            >>> all(inv[sp.S_BOX[i]] == i for i in range(16))
            True
        """
        # [ADD] 新增：计算逆S盒查找表
        inv_s_box = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            inv_s_box[self.S_BOX[i]] = i
        return inv_s_box

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        解密函数

        详细描述：严格按照 small PRESENT-[4] 的逆流程执行：
        - 最后一轮：逆 S 盒 → 轮密钥加
        - 前 rounds-1 轮：逆置换层 → 逆 S 盒 → 轮密钥加
          （按轮次逆序）

        Args:
            ciphertext (np.ndarray): 16 位密文（np.uint8，元素为 0/1）。
            key (np.ndarray): 80 位密钥（np.uint8，元素为 0/1）。

        Returns:
            np.ndarray: 16 位明文（np.uint8，元素为 0/1）。

        Example:
            >>> import numpy as np
            >>> sp = SmallPRESENT4(rounds=4)
            >>> pt = np.random.randint(0, 2, sp.block_size, dtype=np.uint8)
            >>> k = np.random.randint(0, 2, sp.key_size, dtype=np.uint8)
            >>> ct = sp.encrypt(pt, k)
            >>> dec = sp.decrypt(ct, k)
            >>> np.array_equal(pt, dec)
            True
        """
        # 简化：省略类型检查，直接进行逆向流程

        state = ciphertext.copy().astype(np.uint8)
        # 简化：直接调用实例方法
        round_keys = self._key_schedule(key)
        inv_s_box = self._inv_s_box()

        # 最后一轮的逆向：逆S盒 → 轮密钥加
        for nibble_start in range(0, 16, 4):
            nibble = (
                (state[nibble_start] << 3)
                | (state[nibble_start + 1] << 2)
                | (state[nibble_start + 2] << 1)
                | state[nibble_start + 3]
            )
            inv_n = inv_s_box[nibble]
            s_bits = [
                (inv_n >> 3) & 1,
                (inv_n >> 2) & 1,
                (inv_n >> 1) & 1,
                inv_n & 1,
            ]
            for j in range(4):
                state[nibble_start + j] = s_bits[j]
        state ^= round_keys[-1]

        # 前 rounds-1 轮的逆向：逆置换层 → 逆S盒 → 轮密钥加
        for round_idx in range(self.rounds - 2, -1, -1):
            state = self._inv_p_layer(state)
            for nibble_start in range(0, 16, 4):
                nibble = (
                    (state[nibble_start] << 3)
                    | (state[nibble_start + 1] << 2)
                    | (state[nibble_start + 2] << 1)
                    | state[nibble_start + 3]
                )
                inv_n = inv_s_box[nibble]
                s_bits = [
                    (inv_n >> 3) & 1,
                    (inv_n >> 2) & 1,
                    (inv_n >> 1) & 1,
                    inv_n & 1,
                ]
                for j in range(4):
                    state[nibble_start + j] = s_bits[j]
            state ^= round_keys[round_idx]

        return state

    def _key_schedule(self, key: np.ndarray) -> list:
        """论文定义的密钥扩展：生成rounds个16位轮密钥"""
        # 简化：直接访问实例属性
        s_box = self.S_BOX
        rounds = self.rounds
        round_keys = []
        current_key = key.copy().astype(np.uint8)

        for round_idx in range(rounds):
            # 提取前16位作为当前轮密钥
            round_key = current_key[:16].copy()
            round_keys.append(round_key)

            # 1. 密钥左移61位（循环移位）
            # 手动实现循环左移61位（不依赖np.concatenate，避免兼容性问题）
            shift = 61 % 80  # 确保移位量在有效范围内
            rotated = current_key.copy()
            for i in range(80):
                rotated[i] = current_key[(i + shift) % 80]
            current_key = rotated

            # 2. 前4位通过S盒变换
            nibble = (
                (current_key[0] << 3)
                | (current_key[1] << 2)
                | (current_key[2] << 1)
                | current_key[3]
            )
            s_nibble = s_box[nibble]
            current_key[0:4] = [
                (s_nibble >> 3) & 1,
                (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1,
                s_nibble & 1,
            ]

            # 3. 轮常数异或：第 round_idx 轮对第 79-round_idx 位异或 1
            #    round_idx 从 0 开始，且 round_idx < 15
            if round_idx < 15:
                current_key[79 - round_idx] ^= 1

        return round_keys

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        论文定义的加密流程：
        轮密钥加→S盒→置换层（最后一轮无置换）
        """
        # 简化：直接访问实例属性
        s_box = self.S_BOX
        rounds = self.rounds
        state = plaintext.copy().astype(np.uint8)
        # 简化：直接调用实例方法
        round_keys = self._key_schedule(key)

        # 前rounds-1轮：轮密钥加 → S盒替换 → 置换层
        for round_idx in range(rounds - 1):
            state ^= round_keys[round_idx]  # 轮密钥加

            # S盒替换（4bit一组，共4组）
            for nibble_start in range(0, 16, 4):
                nibble = (
                    (state[nibble_start] << 3)
                    | (state[nibble_start + 1] << 2)
                    | (state[nibble_start + 2] << 1)
                    | state[nibble_start + 3]
                )
                s_nibble = s_box[nibble]
                s_bits = [
                    (s_nibble >> 3) & 1,
                    (s_nibble >> 2) & 1,
                    (s_nibble >> 1) & 1,
                    s_nibble & 1,
                ]
                # 逐元素赋值，避免某些 numpy 版本的切片赋值问题
                for j in range(4):
                    state[nibble_start + j] = s_bits[j]

            state = self._p_layer(state)  # 置换层

        # 最后一轮：轮密钥加 → S盒替换（无置换层）
        state ^= round_keys[-1]
        for nibble_start in range(0, 16, 4):
            nibble = (
                (state[nibble_start] << 3)
                | (state[nibble_start + 1] << 2)
                | (state[nibble_start + 2] << 1)
                | state[nibble_start + 3]
            )
            s_nibble = s_box[nibble]
            s_bits = [
                (s_nibble >> 3) & 1,
                (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1,
                s_nibble & 1,
            ]
            for j in range(4):
                state[nibble_start + j] = s_bits[j]

        return state

