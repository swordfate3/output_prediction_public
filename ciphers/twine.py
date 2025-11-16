import numpy as np
from .base_cipher import BaseCipher

class SmallTWINE4(BaseCipher):
    """16位Feistel结构：small TWINE-[4]"""
    def __init__(self, rounds=4):
        super().__init__(block_size=16, key_size=16)
        self.rounds = rounds
        self.branches = 4  # 4个4位分支
        # S盒（同PRESENT，论文2.3节）
        self.S_BOX = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                               0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint8)

    def _f_function(self, x: int, subkey: int) -> int:
        """F函数：子密钥加→S盒替换"""
        return self.S_BOX[x ^ subkey]

    def _key_schedule(self, key: np.ndarray) -> list:
        """生成轮密钥（每轮2个4位子密钥）"""
        round_keys = []
        key_int = int(''.join(map(str, key[:16])), 2)  # 取前16位密钥
        for r in range(self.rounds):
            sk0 = (key_int >> (4 * 3)) & 0xF  # 子密钥0（高4位）
            sk1 = (key_int >> (4 * 1)) & 0xF  # 子密钥1（中间4位）
            round_keys.append((sk0, sk1))
            # 密钥左移8位（循环移位）
            key_int = ((key_int << 8) | (key_int >> 8)) & 0xFFFF
        return round_keys

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Feistel迭代加密"""
        # 拆分为4个4位分支（y0, y1, y2, y3）
        y = [
            (plaintext[0] << 3) | (plaintext[1] << 2) | (plaintext[2] << 1) | plaintext[3],
            (plaintext[4] << 3) | (plaintext[5] << 2) | (plaintext[6] << 1) | plaintext[7],
            (plaintext[8] << 3) | (plaintext[9] << 2) | (plaintext[10] << 1) | plaintext[11],
            (plaintext[12] << 3) | (plaintext[13] << 2) | (plaintext[14] << 1) | plaintext[15]
        ]
        round_keys = self._key_schedule(key)
        for r in range(self.rounds):
            sk0, sk1 = round_keys[r]
            # F函数应用于y0和y1
            f0 = self._f_function(y[0], sk0)
            f1 = self._f_function(y[1], sk1)
            # 分支更新：y2 ^= f0, y3 ^= f1
            y[2] ^= f0
            y[3] ^= f1
            # 轮置换RP：(y0,y1,y2,y3) → (y1,y2,y3,y0)
            y = [y[1], y[2], y[3], y[0]]
        # 合并分支为16位密文
        ciphertext = []
        for branch in y:
            ciphertext.extend([
                (branch >> 3) & 1, (branch >> 2) & 1,
                (branch >> 1) & 1, branch & 1
            ])
        return np.array(ciphertext, dtype=np.uint8)