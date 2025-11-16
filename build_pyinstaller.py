#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller 跨平台构建脚本（Linux/Windows）

提供在项目根目录下执行的标准化打包流程，支持：
- 生成 Linux 可执行文件（需在 Linux 上运行）
- 生成 Windows 可执行文件（需在 Windows 上运行）

[ADD] 构建脚本：统一管理 PyInstaller 命令与常用参数，避免重复配置。
"""

import os
import sys
import subprocess
import shutil
from typing import List, Optional


def detectPlatform() -> str:
    """检测当前运行平台并返回标识

    详细描述：根据 `sys.platform` 判断当前系统类型，返回 `linux`、`windows` 或 `unknown`。

    Args:
        None

    Returns:
        str: 平台标识字符串（`linux`、`windows`、`unknown`）

    Raises:
        None

    Example:
        >>> detectPlatform()
        'linux'
    """
    plat = sys.platform
    if plat.startswith("linux"):
        return "linux"
    if plat.startswith("win"):
        return "windows"
    return "unknown"


def resolveEntry(default_entry: str = "main.py") -> str:
    """解析并返回入口脚本路径

    详细描述：优先使用传入的默认入口 `main.py`，若不存在则报错提示。用户可修改为自定义入口。

    Args:
        default_entry (str): 默认入口脚本文件路径，默认为 `main.py`

    Returns:
        str: 入口脚本的有效路径

    Raises:
        FileNotFoundError: 当入口脚本文件不存在时抛出

    Example:
        >>> resolveEntry()
        'main.py'
    """
    if not os.path.exists(default_entry):
        raise FileNotFoundError(f"未找到入口脚本: {default_entry}，请确认路径或修改脚本配置")
    return default_entry


def ensureOutputDir(dir_path: str) -> None:
    """确保输出目录存在并可写

    详细描述：若目录不存在则创建，存在则保持不变。用于存放打包后的产物。

    Args:
        dir_path (str): 目标目录路径

    Returns:
        None

    Raises:
        Exception: 创建目录失败（权限或文件系统错误）

    Example:
        >>> ensureOutputDir('dist')
    """
    os.makedirs(dir_path, exist_ok=True)


def buildLinux(entry: str, onefile: bool = True, name: Optional[str] = None, extra_args: Optional[List[str]] = None) -> None:
    """在 Linux 上构建可执行文件

    详细描述：调用 PyInstaller 生成 Linux 平台的可执行文件。默认使用 `--onefile` 单文件模式，可通过 `extra_args` 追加自定义参数。

    Args:
        entry (str): 入口脚本路径（例如 `main.py`）
        onefile (bool): 是否使用单文件模式，默认为 True
        name (Optional[str]): 生成的可执行文件名称，默认为 `OutputPrediction-linux`
        extra_args (Optional[List[str]]): 追加的 PyInstaller 参数列表

    Returns:
        None

    Raises:
        RuntimeError: 当当前平台不是 Linux 时抛出
        FileNotFoundError: 当入口脚本不存在时抛出

    Example:
        >>> buildLinux(entry='main.py', onefile=True)
    """
    if detectPlatform() != "linux":
        raise RuntimeError("当前平台不是 Linux，无法构建 Linux 版可执行文件")
    entry_path = resolveEntry(entry)

    target_name = name or "OutputPrediction-linux"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--name",
        target_name,
    ]
    if onefile:
        cmd.append("--onefile")

    # [ADD] 排除不需要的目录（作为说明；PyInstaller默认不会打包未导入的代码/资源）
    # 如需强制排除某些模块，可追加 --exclude-module 选项
    excludes = [
        "--exclude-module", "tests",
        "--exclude-module", "examples",
    ]
    cmd.extend(excludes)

    # 追加用户自定义参数
    if extra_args:
        cmd.extend(extra_args)

    # 指定入口脚本
    cmd.append(entry_path)

    print("开始构建 Linux 可执行文件:")
    print(" ", " ".join(cmd))
    ensureOutputDir("dist")
    subprocess.check_call(cmd)
    print("✓ Linux 构建完成，产物位于 dist/ 目录")


def buildWindows(entry: str, onefile: bool = True, name: Optional[str] = None, extra_args: Optional[List[str]] = None) -> None:
    """在 Windows 上构建可执行文件

    详细描述：调用 PyInstaller 生成 Windows 平台的可执行文件。注意：PyInstaller 不支持跨平台构建，必须在 Windows 系统上运行本函数。

    Args:
        entry (str): 入口脚本路径（例如 `main.py`）
        onefile (bool): 是否使用单文件模式，默认为 True
        name (Optional[str]): 生成的可执行文件名称，默认为 `OutputPrediction-windows`
        extra_args (Optional[List[str]]): 追加的 PyInstaller 参数列表

    Returns:
        None

    Raises:
        RuntimeError: 当当前平台不是 Windows 时抛出
        FileNotFoundError: 当入口脚本不存在时抛出

    Example:
        >>> buildWindows(entry='main.py', onefile=True)
    """
    if detectPlatform() != "windows":
        raise RuntimeError("当前平台不是 Windows，无法构建 Windows 版可执行文件")
    entry_path = resolveEntry(entry)

    target_name = name or "OutputPrediction-windows"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--name",
        target_name,
    ]
    if onefile:
        cmd.append("--onefile")

    # [ADD] Windows 下常用的收集参数，可用于提升兼容性（如 torch、numpy）
    common_collects = [
        "--collect-all", "torch",
        "--collect-submodules", "numpy",
    ]
    cmd.extend(common_collects)

    # 追加用户自定义参数
    if extra_args:
        cmd.extend(extra_args)

    # 指定入口脚本
    cmd.append(entry_path)

    print("开始构建 Windows 可执行文件:")
    print(" ", " ".join(cmd))
    ensureOutputDir("dist")
    subprocess.check_call(cmd)
    print("✓ Windows 构建完成，产物位于 dist/ 目录")


def cleanDist() -> None:
    """清理打包产物目录

    详细描述：删除 `build/` 与 `dist/` 目录以及临时的 `*.spec` 文件，确保重新构建的干净环境。

    Args:
        None

    Returns:
        None

    Raises:
        Exception: 当删除目录/文件失败时可能抛出

    Example:
        >>> cleanDist()
    """
    for d in ("build", "dist"):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for f in os.listdir("."):
        if f.endswith(".spec"):
            try:
                os.remove(f)
            except Exception:
                pass
    print("✓ 已清理 build/、dist/ 与临时 .spec 文件")


def main() -> None:
    """命令行入口：解析参数并触发对应平台构建

    详细描述：支持参数：
    - `--target {linux,windows}` 指定目标平台
    - `--entry PATH` 指定入口脚本（默认 `main.py`）
    - `--onefile/--no-onefile` 控制单文件模式
    - `--clean` 清理产物后再构建

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: 参数不合法时可能由 argparse 抛出

    Example:
        >>> # 在 Linux 打包
        >>> # python build_pyinstaller.py --target linux --onefile
        >>> # 在 Windows 打包
        >>> # python build_pyinstaller.py --target windows --onefile
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyInstaller 跨平台构建脚本")
    parser.add_argument("--target", choices=["linux", "windows"], required=True, help="目标平台")
    parser.add_argument("--entry", default="main.py", help="入口脚本路径，默认 main.py")
    parser.add_argument("--onefile", action="store_true", help="启用单文件模式")
    parser.add_argument("--no-onefile", dest="onefile", action="store_false", help="禁用单文件模式")
    parser.add_argument("--clean", action="store_true", help="构建前清理 build/dist 目录与 .spec 文件")

    args = parser.parse_args()

    if args.clean:
        cleanDist()

    if args.target == "linux":
        buildLinux(entry=args.entry, onefile=args.onefile)
    elif args.target == "windows":
        buildWindows(entry=args.entry, onefile=args.onefile)
    else:
        print("未知目标平台")


if __name__ == "__main__":
    main()