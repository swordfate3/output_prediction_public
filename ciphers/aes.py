import numpy as np
from .base_cipher import BaseCipher


class SmallAES4(BaseCipher):
    """16位AES-like SPN结构：small AES-[4]"""

    def __init__(self, rounds=4):
        super().__init__(block_size=16, key_size=16)
        self.rounds = rounds
        # 与PRESENT相同的S盒（论文2.2节）
        self.S_BOX = np.array(
            [
                0xC, 0x5, 0x6, 0xB,
                0x9, 0x0, 0xA, 0xD,
                0x3, 0xE, 0xF, 0x8,
                0x4, 0x7, 0x1, 0x2
            ],
            dtype=np.uint8,
        )
        # MDS矩阵（论文公式）
        self.MDS = np.array(
            [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]], dtype=np.uint8
        )

    def _mix_columns(self, state: np.ndarray) -> np.ndarray:
        """列混合：GF(2^4)上的MDS矩阵乘法"""
        # 将16位状态重新组织为4个4位值
        values = []
        for i in range(4):
            val = 0
            for j in range(4):
                val |= state[i * 4 + j] << (3 - j)
            values.append(val)

        # 应用MDS矩阵乘法
        new_values = []
        for i in range(4):
            result = 0
            for j in range(4):
                # GF(2^4)乘法，使用不可约多项式x^4+x+1
                product = self._gf4_multiply(self.MDS[i][j], values[j])
                result ^= product
            new_values.append(result)

        # 转换回16位状态
        new_state = np.zeros(16, dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                new_state[i * 4 + j] = (new_values[i] >> (3 - j)) & 1

        return new_state

    def _gf4_multiply(self, a: int, b: int) -> int:
        """GF(2^4)域上的乘法运算，使用不可约多项式x^4+x+1"""
        result = 0
        while b > 0:
            if b & 1:
                result ^= a
            a <<= 1
            if a & 0x10:  # 如果超出4位
                a ^= 0x13  # 模x^4+x+1 (0x13 = 0b10011)
            b >>= 1
        return result & 0xF

    def _key_schedule(self, key: np.ndarray) -> list:
        """适用于16位密钥的密钥扩展算法"""
        round_keys = []
        current_key = key.copy().astype(np.uint8)
        for r in range(self.rounds):
            round_key = current_key[:16]
            round_keys.append(round_key)
            # 16位密钥的简化扩展：循环左移4位
            temp_key = np.zeros_like(current_key, dtype=np.uint8)
            key_len = len(current_key)
            for i in range(key_len):
                temp_key[i] = current_key[(i + 4) % key_len]
            current_key = temp_key
            # 前4位通过S盒
            nibble = (
                (current_key[0] << 3)
                | (current_key[1] << 2)
                | (current_key[2] << 1)
                | current_key[3]
            )
            s_nibble = self.S_BOX[nibble]
            # 逐个元素赋值，避免切片赋值的兼容性问题
            s_bits = [
                (s_nibble >> 3) & 1,
                (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1,
                s_nibble & 1,
            ]
            for j in range(4):
                current_key[j] = s_bits[j]
            # 轮常数异或（适用于16位密钥）
            if r < 15:
                current_key[int(15 - r)] ^= 1
        return round_keys

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        state = plaintext.copy().astype(np.uint8)
        round_keys = self._key_schedule(key)
        for r in range(self.rounds - 1):
            state ^= round_keys[r]  # 轮密钥加
            # S盒替换
            for i in range(4):
                nibble = (
                    (state[i * 4] << 3)
                    | (state[i * 4 + 1] << 2)
                    | (state[i * 4 + 2] << 1)
                    | state[i * 4 + 3]
                )
                s_nibble = self.S_BOX[nibble]
                # 逐个元素赋值，避免切片赋值的兼容性问题
                s_bits = [
                    (s_nibble >> 3) & 1,
                    (s_nibble >> 2) & 1,
                    (s_nibble >> 1) & 1,
                    s_nibble & 1,
                ]
                for j in range(4):
                    state[i * 4 + j] = s_bits[j]
            state = self._mix_columns(state)  # 列混合（替代PRESENT的置换层）
        # 最后一轮（无列混合）
        state ^= round_keys[-1]
        for i in range(4):
            nibble = (
                (state[i * 4] << 3)
                | (state[i * 4 + 1] << 2)
                | (state[i * 4 + 2] << 1)
                | state[i * 4 + 3]
            )
            s_nibble = self.S_BOX[nibble]
            # 逐个元素赋值，避免切片赋值的兼容性问题
            s_bits = [
                (s_nibble >> 3) & 1,
                (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1,
                s_nibble & 1,
            ]
            for j in range(4):
                state[i * 4 + j] = s_bits[j]
        return state

    def _inv_mix_columns(self, state: np.ndarray) -> np.ndarray:
        """逆列混合 - GF(2^4)上的逆MDS矩阵乘法"""
        # 逆MDS矩阵
        inv_mds = np.array(
            [[14, 11, 13, 9], [9, 14, 11, 13], [13, 9, 14, 11], [11, 13, 9, 14]], dtype=np.uint8
        )
        
        # 将16位状态重新组织为4个4位值
        values = []
        for i in range(4):
            val = 0
            for j in range(4):
                val |= state[i * 4 + j] << (3 - j)
            values.append(val)

        # 应用逆MDS矩阵乘法
        new_values = []
        for i in range(4):
            result = 0
            for j in range(4):
                # GF(2^4)乘法，使用不可约多项式x^4+x+1
                product = self._gf4_multiply(inv_mds[i][j], values[j])
                result ^= product
            new_values.append(result)

        # 转换回16位状态
        new_state = np.zeros(16, dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                new_state[i * 4 + j] = (new_values[i] >> (3 - j)) & 1

        return new_state

    def _inv_s_box(self) -> np.ndarray:
        """逆S盒"""
        inv_s_box = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            inv_s_box[self.S_BOX[i]] = i
        return inv_s_box

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        解密函数 - 逆向执行加密步骤
        """
        state = ciphertext.copy().astype(np.uint8)
        round_keys = self._key_schedule(key)
        inv_s_box = self._inv_s_box()
        
        # 最后一轮的逆向：逆S盒 -> 轮密钥加
        for i in range(4):
            nibble = (
                (state[i * 4] << 3)
                | (state[i * 4 + 1] << 2)
                | (state[i * 4 + 2] << 1)
                | state[i * 4 + 3]
            )
            inv_nibble = inv_s_box[nibble]
            s_bits = [
                (inv_nibble >> 3) & 1,
                (inv_nibble >> 2) & 1,
                (inv_nibble >> 1) & 1,
                inv_nibble & 1,
            ]
            for j in range(4):
                state[i * 4 + j] = s_bits[j]
        
        state ^= round_keys[-1]
        
        # 前rounds-1轮的逆向：逆列混合 -> 逆S盒 -> 轮密钥加
        for round_idx in range(self.rounds - 2, -1, -1):
            state = self._inv_mix_columns(state)
            
            # 逆S盒
            for i in range(4):
                nibble = (
                    (state[i * 4] << 3)
                    | (state[i * 4 + 1] << 2)
                    | (state[i * 4 + 2] << 1)
                    | state[i * 4 + 3]
                )
                inv_nibble = inv_s_box[nibble]
                s_bits = [
                    (inv_nibble >> 3) & 1,
                    (inv_nibble >> 2) & 1,
                    (inv_nibble >> 1) & 1,
                    inv_nibble & 1,
                ]
                for j in range(4):
                    state[i * 4 + j] = s_bits[j]
            
            state ^= round_keys[round_idx]
        
        return state
