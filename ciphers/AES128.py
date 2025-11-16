import numpy as np
from .base_cipher import BaseCipher


class AES128(BaseCipher):
    """完整 AES-128 加密实现（位数组接口，轮数可配置）

    - 明文：长度 128 的位数组（np.uint8，0/1）
    - 密钥：长度 128 的位数组（np.uint8，0/1）
    - 轮数：默认10，最多支持14
    """
    def __init__(self, rounds: int = 10):
        super().__init__(block_size=128, key_size=128)
        self.rounds = int(rounds)
        self.S_BOX = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
            0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
            0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
            0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
            0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
            0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
            0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
            0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
            0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
            0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
            0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
            0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
            0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
            0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
            0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
            0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
            0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
        ], dtype=np.uint8)
        self.RCON = np.array(
            [
                0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
                0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D,
            ],
            dtype=np.uint8,
        )

    # ===== 位/字节转换 =====
    @staticmethod
    def _bytes_from_bits(bits: np.ndarray) -> np.ndarray:
        if bits.dtype != np.uint8:
            bits = bits.astype(np.uint8)
        if bits.ndim != 1 or bits.size != 128:
            raise ValueError("AES128 需要长度为128的一维位数组")
        out = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            b = 0
            base = i * 8
            for j in range(8):
                b |= (bits[base + j] & 1) << (7 - j)
            out[i] = b
        return out

    @staticmethod
    def _bits_from_bytes(bytes_arr: np.ndarray) -> np.ndarray:
        out = np.zeros(128, dtype=np.uint8)
        for i in range(16):
            b = int(bytes_arr[i]) & 0xFF
            base = i * 8
            for j in range(8):
                out[base + j] = (b >> (7 - j)) & 1
        return out

    # ===== 组件 =====
    def _sub_bytes(self, state: np.ndarray) -> np.ndarray:
        return self.S_BOX[state]

    @staticmethod
    def _shift_rows(state: np.ndarray) -> np.ndarray:
        mat = [[0]*4 for _ in range(4)]
        for c in range(4):
            for r in range(4):
                mat[r][c] = int(state[4*c + r])
        for r in range(4):
            mat[r] = mat[r][r:] + mat[r][:r]
        out = np.zeros(16, dtype=np.uint8)
        for c in range(4):
            for r in range(4):
                out[4*c + r] = mat[r][c]
        return out

    @staticmethod
    def _xtime(b: int) -> int:
        b <<= 1
        if b & 0x100:
            b ^= 0x11B
        return b & 0xFF

    @classmethod
    def _mul2(cls, b: int) -> int:
        return cls._xtime(b)

    @classmethod
    def _mul3(cls, b: int) -> int:
        return cls._mul2(b) ^ (b & 0xFF)

    @classmethod
    def _mix_columns(cls, state: np.ndarray) -> np.ndarray:
        out = np.zeros(16, dtype=np.uint8)
        for c in range(4):
            s0 = int(state[4*c + 0])
            s1 = int(state[4*c + 1])
            s2 = int(state[4*c + 2])
            s3 = int(state[4*c + 3])
            out[4*c + 0] = cls._mul2(s0) ^ cls._mul3(s1) ^ s2 ^ s3
            out[4*c + 1] = s0 ^ cls._mul2(s1) ^ cls._mul3(s2) ^ s3
            out[4*c + 2] = s0 ^ s1 ^ cls._mul2(s2) ^ cls._mul3(s3)
            out[4*c + 3] = cls._mul3(s0) ^ s1 ^ s2 ^ cls._mul2(s3)
        return out

    @staticmethod
    def _add_round_key(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(state, round_key)

    # ===== 密钥扩展 =====
    def _rot_word(self, word: np.ndarray) -> np.ndarray:
        return np.array([word[1], word[2], word[3], word[0]], dtype=np.uint8)

    def _sub_word(self, word: np.ndarray) -> np.ndarray:
        return self.S_BOX[word]

    def _key_schedule(self, key_bits: np.ndarray) -> list:
        key_bytes = self._bytes_from_bits(key_bits)
        Nk = 4
        Nr = max(1, min(self.rounds, 14))
        Nb = 4
        total_words = Nb * (Nr + 1)

        w = [
            np.array(key_bytes[4*i:4*(i+1)], dtype=np.uint8)
            for i in range(Nk)
        ]
        i = Nk
        while len(w) < total_words:
            temp = w[-1].copy()
            if i % Nk == 0:
                temp = self._sub_word(self._rot_word(temp)).copy()
                temp[0] ^= int(self.RCON[i // Nk] & 0xFF)
            w.append(np.bitwise_xor(w[i - Nk], temp))
            i += 1

        round_keys = []
        for r in range(Nr + 1):
            rk = np.concatenate(w[r*Nb:(r+1)*Nb])
            round_keys.append(rk.astype(np.uint8))
        return round_keys

    # ===== 加密 =====
    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        if plaintext.size != 128 or key.size != 128:
            raise ValueError("AES128 仅支持128位明文与128位密钥")
        state = self._bytes_from_bits(plaintext)
        round_keys = self._key_schedule(key)
        Nr = len(round_keys) - 1

        state = self._add_round_key(state, round_keys[0])

        for r in range(1, Nr):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, round_keys[r])

        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, round_keys[Nr])

        return self._bits_from_bytes(state)