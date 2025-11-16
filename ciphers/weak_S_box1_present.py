from .present import SmallPRESENT4
import numpy as np
class SmallPRESENT4_WeakSBox1(SmallPRESENT4):
    """使用weak S-box1的small PRESENT-[4]（论文表3）"""
    def __init__(self, rounds=4):
        super().__init__(rounds=rounds)
        # 弱S盒1：对差分攻击脆弱（论文表3）
        self.S_BOX = np.array([
            0x6, 0x4, 0xC, 0x5, 0x0, 0x7, 0x2, 0xE,
            0x1, 0xF, 0x3, 0xD, 0x8, 0xA, 0x9, 0xB
        ], dtype=np.uint8)
        # 重新计算逆S盒
        self.S_BOX_INV = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            self.S_BOX_INV[int(self.S_BOX[i])] = i