# -*- coding: utf-8 -*-
"""
ciphers 包初始化文件

这个包包含了各种密码算法的实现，包括：
- AES 密码算法
- PRESENT 密码算法及其变种
- TWINE 密码算法
- 基础密码类和组件修改版本

Author: Output Prediction Project
"""

# 导入主要的密码类，方便外部使用
try:
    from .present import SmallPRESENT4
    from .aes import SmallAES4
    from .AES128 import AES128
    from .AES64 import AES64
    from .grain128a import Grain128a
    from .twine import SmallTWINE4
    from .base_cipher import BaseCipher
    from .weak_S_box1_present import SmallPRESENT4_WeakSBox1
    from .weak_S_box2_present import SmallPRESENT4_WeakSBox2
    from .component_modification_present import SmallPRESENT4_SwapComponents
    # [ADD] 新增: 导入 Trivium 流密码实现
    from .trivium import Trivium
    # [ADD] 新增: 导入 AcornV3 流密码（工程简化版接口）
    from .acornv3 import AcornV3
    # [ADD] 新增: 导入 Zuc256 流密码（研究用途简化实现）
    from .zuc256 import Zuc256
except ImportError as e:
    # 如果某些模块导入失败，记录但不中断整个包的加载
    import warnings
    warnings.warn(f"部分密码模块导入失败: {e}", ImportWarning)

__version__ = "1.0.0"
__author__ = "Output Prediction Project"
__all__ = [
    "SmallPRESENT4",
    "SmallAES4", 
    "AES128",
    "AES64",
    "Grain128a",
    "SmallTWINE4",
    "BaseCipher",
    "SmallPRESENT4_WeakSBox1",
    "SmallPRESENT4_WeakSBox2",
    "SmallPRESENT4_SwapComponents",
    # [ADD] 新增: 将 Trivium 加入导出列表
    "Trivium",
    # [ADD] 新增: 将 AcornV3 加入导出列表
    "AcornV3",
    # [ADD] 新增: 将 Zuc256 加入导出列表
    "Zuc256",
]