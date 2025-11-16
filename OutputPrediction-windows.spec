# -*- mode: python ; coding: utf-8 -*-
# [MOD] 说明：仿照 pyinstaller-linux.spec 的结构，恢复函数式配置
#             并在 Windows 构建中按需复制 configs/config.json 到 dist。

import os  # [ADD]

block_cipher = None


def hookspath():
    """返回自定义钩子路径列表

    详细描述：Windows 下默认不使用自定义钩子，返回空列表。

    Args:
        None

    Returns:
        list: 钩子路径列表

    Raises:
        None

    Example:
        >>> hookspath()
        []
    """
    return []


def datas():
    """返回数据文件映射列表

    详细描述：若项目根目录存在 `configs/config.json`，则在打包时复制到
    `dist/OutputPrediction-windows/configs/`，以便构建后可直接修改该文件
    实现外部配置覆盖（与 `utils.config.Config.findRuntimeConfigPath` 配合）。

    Args:
        None

    Returns:
        list[tuple]: 形如 `(源路径, 目标目录)` 的数据映射

    Raises:
        None

    Example:
        >>> datas()
        [('configs/config.json', 'configs')]
    """
    items = []
    cfg_path = os.path.join('configs', 'config.json')
    if os.path.exists(cfg_path):
        # [ADD] 若存在则复制到 dist 下的 configs 目录，构建后可编辑
        items.append((cfg_path, 'configs'))
    return items


def binaries():
    """返回二进制依赖映射列表

    详细描述：默认不添加，保持最简配置；如需添加 DLL/so 等可在此扩展。

    Args:
        None

    Returns:
        list[tuple]: 二进制文件映射

    Raises:
        None

    Example:
        >>> binaries()
        []
    """
    return []


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries(),
    datas=datas(),  # [ADD] 复制外部 config.json 以支持构建后修改
    hiddenimports=[],
    hookspath=hookspath(),
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tests', 'examples','md','logs','data','papers','results'],  # [ADD] 排除不必要目录
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OutputPrediction-windows',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OutputPrediction-windows',
)
