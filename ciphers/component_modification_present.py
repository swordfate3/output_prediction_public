import numpy as np
from .present import SmallPRESENT4  # 继承自基础PRESENT实现

class SmallPRESENT4_SwapComponents(SmallPRESENT4):
    """组件顺序修改的small PRESENT-[4]：先置换层后S盒"""
    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        state = plaintext.copy().astype(np.uint8)
        round_keys = self._key_schedule(key)
        
        for r in range(self.rounds - 1):
            # 轮密钥加（与原始一致）
            state ^= round_keys[r]
            # 核心修改：先置换层，再S盒（原始为S盒→置换层）
            state = self._p_layer(state)  # 置换层提前
            # S盒替换（4位一组）
            for i in range(4):
                nibble = (state[i*4] << 3) | (state[i*4+1] << 2) | (state[i*4+2] << 1) | state[i*4+3]
                s_nibble = self.S_BOX[nibble]
                # 逐个元素赋值，避免切片赋值的兼容性问题
                s_bits = [
                    (s_nibble >> 3) & 1, (s_nibble >> 2) & 1,
                    (s_nibble >> 1) & 1, s_nibble & 1
                ]
                for j in range(4):
                    state[i*4+j] = s_bits[j]
        
        # 最后一轮（无置换层，保持修改后顺序）
        state ^= round_keys[-1]
        state = self._p_layer(state)  # 置换层提前
        for i in range(4):
            nibble = (state[i*4] << 3) | (state[i*4+1] << 2) | (state[i*4+2] << 1) | state[i*4+3]
            s_nibble = self.S_BOX[nibble]
            # 逐个元素赋值，避免切片赋值的兼容性问题
            s_bits = [
                (s_nibble >> 3) & 1, (s_nibble >> 2) & 1,
                (s_nibble >> 1) & 1, s_nibble & 1
            ]
            for j in range(4):
                state[i*4+j] = s_bits[j]
        
        return state