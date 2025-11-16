# PyInstaller Spec 文件详解（Windows 与 Linux 通用）

> 这是一份专注于 `.spec` 文件的实用教程，结合当前项目 `Output_Prediction` 的结构与已有 `pyinstaller-windows.spec`、`pyinstaller-linux.spec` 文件，帮助你快速理解与定制打包配置。

---

## 为什么需要 `.spec` 文件
- `.spec` 是 PyInstaller 的构建脚本，描述了如何收集代码、依赖、资源并生成最终的可执行文件。
- 对比命令行参数，`.spec` 更适合复杂工程：可维护、可复用、可版本化。

## Spec 文件的核心结构
- `Analysis`：静态分析入口文件与依赖。
- `PYZ`：将 Python 源码打包为 `pyz` 存档。
- `EXE`：生成可执行文件（`onefile` 模式下即最终产物）。
- `COLLECT`：收集运行时所需的所有文件（`onedir` 模式）。
- 平台差异：macOS 还可能看到 `BUNDLE`；Windows/Linux 使用 `EXE` 与 `COLLECT` 为主。

### 关键参数速览
- `datas`：要打包的非代码资源（图片、字体、配置等）。
- `binaries`：本地二进制依赖（`.so`/`.dll`）。
- `hiddenimports`：动态导入或分析器无法静态推断的模块名。
- `excludes`：明确排除的模块或包。
- `name`、`console`、`icon`、`upx`、`strip`：产物名、是否显示控制台、图标、压缩与瘦身。

---

## 项目中的两个 spec 文件说明
- `pyinstaller-windows.spec`：用于 Windows 产物，常见差异是 `.dll`、图标、是否显示控制台等；图标可通过 `icon='assets/app.ico'` 指定。
- `pyinstaller-linux.spec`：用于 Linux 产物，你当前文件中存在如下行：

  ```python
  excludes=['tests', 'examples','md','logs','data'],  # [ADD] 排除不需要的目录模块
  ```
  - 含义：打包时不包含 `tests/examples/md/logs/data` 目录下的模块，减小体积、避免不必要文件进入产物。
  - 如需保留某目录（例如 `data`），可删除或调整该项（见后文的删除示例）。

---

## 在本项目中的常用配置示例

### 1) 添加数据文件到产物（字体、图片、配置等）
```python
# [ADD] 示例：将 assets/fonts 下的所有字体文件打包，并在运行时路径为 fonts/
# 说明：datas 元组格式为 (源文件或通配符, 目标子目录)
datas=[('assets/fonts/*.ttf', 'fonts')]
```

### 2) 精简体积与运行体验
```python
# [ADD] 示例：在 EXE 中启用 strip 与 upx（若系统安装了 upx）
# 注意：upx 在某些平台或依赖上可能导致问题，使用前请先验证
exe = EXE(
    # ... 其他参数 ...
    strip=True,
    upx=True,
)
```

### 3) 指定产物名称与是否显示控制台窗口
```python
# [ADD] 示例：命名产物并在 Windows 上隐藏控制台（用于 GUI/纯日志文件场景）
exe = EXE(
    # ... 其他参数 ...
    name='Output_Prediction',
    console=False,  # Windows 可隐藏控制台；Linux 通常保留 True 便于调试
)
```

### 4) 添加隐藏导入（解决动态导入、插件式架构）
```python
# [ADD] 示例：当使用 importlib/dynamic import 或某些库的子模块未被静态分析到时
hiddenimports=[
    'matplotlib', 'matplotlib.pyplot',
    'sklearn', 'sklearn.metrics',
    'numpy',
    # 若使用 torch 或其他框架的动态插件，这里添加其子模块
]
```

---

## 路径与资源访问的正确方式
打包后，资源文件的真实路径可能不同。推荐使用以下辅助函数来兼容开发态与打包态。

```python
def getResourcePath(relative_path: str) -> str:
    """
    获取资源文件的绝对路径，兼容 PyInstaller 打包与普通开发环境。

    在使用 PyInstaller 的 onefile/onedir 模式下，资源可能被展开在临时目录或 dist 子目录。
    本函数通过检测 `sys._MEIPASS`（PyInstaller 运行时临时目录）来定位资源文件。

    Args:
        relative_path (str): 相对资源路径（相对于工程根或打包时设定的子目录）。

    Returns:
        str: 资源文件的绝对路径，保证在开发与打包两种环境下均可访问。

    Raises:
        FileNotFoundError: 当目标资源文件不存在时抛出。

    Example:
        >>> path = getResourcePath('fonts/simhei.ttf')
        >>> print(path)
    """
    import os
    import sys

    # [ADD] 优先从 PyInstaller 的临时目录定位资源
    base_path = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    abs_path = os.path.join(base_path, relative_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f'Resource not found: {abs_path}')
    return abs_path
```

```python
def buildDatas(root_dir: str) -> list:
    """
    根据给定根目录动态收集数据文件，生成适用于 PyInstaller `datas` 的列表。

    支持通配符与子目录映射，便于集中管理非代码资源（如字体、图片、配置）。

    Args:
        root_dir (str): 项目中资源根目录，如 'assets'。

    Returns:
        list: 形如 [(源, 目标子目录), ...] 的列表，可直接传入 Analysis 的 `datas` 参数。

    Raises:
        ValueError: 当根目录不存在或不可读时抛出异常。

    Example:
        >>> assets = buildDatas('assets')
        >>> # 结合 spec 使用：datas=assets
    """
    import os
    import glob

    # [ADD] 校验根目录并收集常见类型资源
    if not os.path.isdir(root_dir):
        raise ValueError(f'Invalid assets root: {root_dir}')

    patterns = [
        ('fonts/*.ttf', 'fonts'),
        ('images/*.*', 'images'),
        ('configs/*.yml', 'configs'),
    ]

    results = []
    for pattern, target in patterns:
        for src in glob.glob(os.path.join(root_dir, pattern)):
            results.append((src, target))
    return results
```

> 以上两个函数是示例辅助代码，放置到你项目的某个模块（如 `utils/directory_manager.py`）后，即可在 `.spec` 中导入使用；它们包含了函数级注释，便于理解与维护。

---

## 最小可用的 spec 模板（含注释）
```python
# -*- mode: python ; coding: utf-8 -*-

# [ADD] 基本的 Spec 模板，适用于大多数 CLI/脚本类项目
import sys
from PyInstaller.utils.hooks import collect_submodules

# [ADD] 按需收集隐藏子模块（示例）
hidden = collect_submodules('matplotlib')

# [ADD] 入口脚本（你的项目为 main.py）
block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/fonts/*.ttf', 'fonts')],  # [ADD] 示例添加数据文件
    hiddenimports=hidden,                      # [ADD] 示例添加隐藏导入
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tests', 'examples', 'md', 'logs', 'data'],  # [ADD] 精简体积示例
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Output_Prediction',     # [ADD] 指定产物名
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,                   # [ADD] 瘦身
    upx=True,                     # [ADD] 压缩（需本机安装 upx）
    console=True,                 # [ADD] 控制台显示，Windows 可改为 False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Output_Prediction',
)
```

### 删除/调整排除项的示例
```python
# [DEL] 如需保留 data 目录中的资源到产物中，可删除 'data'
excludes=['tests', 'examples', 'md', 'logs']
```

---

## Windows 与 Linux 的常见差异
- 字体与中文显示：Windows 下字体通常在系统可用；Linux 下建议随包附带字体（如 `SimHei`），并在程序中显式设置。
- 二进制依赖：Windows 常为 `.dll`，Linux 常为 `.so`，必要时写入 `binaries`。
- 控制台：Windows GUI 程序常用 `console=False`；Linux 为调试便利可设为 `True`。
- 图标：Windows 使用 `icon='.ico'`，Linux 多数桌面环境不使用 exe 图标（可忽略）。

---

## 打包命令与步骤
- 安装依赖：
  - `pip install -r requirements.txt`
  - `pip install pyinstaller`
- Linux 构建：
  - `pyinstaller pyinstaller-linux.spec`
- Windows 构建：
  - `pyinstaller pyinstaller-windows.spec`
- 清理缓存后重打包：
  - `pyinstaller --clean pyinstaller-linux.spec`
- 产物位置：
  - `dist/Output_Prediction/`（onedir）或 `dist/Output_Prediction`（onefile，可执行文件）

---

## 调试与常见问题
- 增加日志：`pyinstaller --log-level DEBUG pyinstaller-linux.spec`
- 动态导入缺失：使用 `hiddenimports` 或 `collect_submodules` 收集子模块。
- 体积过大：
  - 使用 `excludes` 排除不需要的包与目录。
  - 启用 `strip`/`upx`（在验证兼容性的前提下）。
- 相对路径失效：使用前文的 `getResourcePath`。

---

## 与当前项目结构的适配建议
- 入口脚本：`main.py`（确保为 Analysis 的入口）。
- 目录建议：
  - 不打包 `tests/`、`examples/`、`md/`、`logs/`、`results/`、`plots/`（除非程序运行时确需读取）。
  - 若程序需要运行时读取模型或特定配置，请用 `datas` 显式添加。
- 可能的隐藏导入：
  - `matplotlib`、`sklearn`、`numpy`、`torch`、项目下的 `ciphers/*` 等，如遇报错按需加入。

---

## 快速检查清单（Cheat Sheet）
- 修改入口：`Analysis(['main.py'])`。
- 添加资源：`datas=[('assets/fonts/*.ttf', 'fonts')]`。
- 定义隐藏导入：`hiddenimports=[...]` 或 `collect_submodules('package')`。
- 精简体积：`excludes=[...]`；`strip=True`；`upx=True`。
- 控制台/图标：`console=False`（Windows GUI）；`icon='assets/app.ico'`。
- 运行时路径：使用 `getResourcePath()` 获取资源文件。

---

> 若需要，我可以根据你当前的 `pyinstaller-windows.spec` 与 `pyinstaller-linux.spec` 进行逐行注释与优化，或提供一个适配你依赖的完整版本（含 `datas/hiddenimports` 的精细清单）。