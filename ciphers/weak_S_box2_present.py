from .present import SmallPRESENT4
import numpy as np
class SmallPRESENT4_WeakSBox2(SmallPRESENT4):
    """使用weak S-box2的small PRESENT-[4]（论文表4）"""
    def __init__(self, rounds=4):
        super().__init__(rounds=rounds)
        # 弱S盒2：对线性攻击脆弱（论文表4）
        self.S_BOX = np.array([
            0xF, 0xE, 0xB, 0xC, 0x6, 0xD, 0x7, 0x8,
            0x0, 0x3, 0x9, 0xA, 0x4, 0x2, 0x1, 0x5
        ], dtype=np.uint8)
        # 重新计算逆S盒
        self.S_BOX_INV = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            self.S_BOX_INV[int(self.S_BOX[i])] = i