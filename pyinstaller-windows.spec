# -*- mode: python ; coding: utf-8 -*-
# [ADD] Windows 版 PyInstaller spec 文件：配置单文件构建与常见依赖收集

block_cipher = None


def hookspath():
    return []


def datas():
    # [ADD] 可按需添加数据文件映射，例如 (src, dest)
    return []


def binaries():
    # [ADD] 可按需添加二进制依赖
    return []


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries(),
    datas=datas(),
    hiddenimports=['numpy'],  # [ADD] 常见的隐式依赖
    hookspath=hookspath(),
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tests', 'examples','md','logs','data','papers','results'],  # [ADD] 排除不需要的目录模块
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