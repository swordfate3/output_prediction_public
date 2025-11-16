# Output_Prediction 项目打包指南（PyInstaller）

本指南介绍如何在 Linux 与 Windows 上使用 PyInstaller 将项目打包为可执行文件（`exe`/ELF）。已提供标准构建脚本与 spec 文件，支持单文件模式，默认入口为 `main.py`。

## 目录
- 前置准备
- 快速打包命令
- 使用构建脚本（推荐）
- 使用 spec 文件（高级控制）
- 常见问题与排除项

## 前置准备
- 安装依赖：
  - `pip install -r requirements.txt`
  - `pip install pyinstaller`
- 选择入口：默认使用 `main.py`。如需更换入口，请修改命令或脚本参数。
- 注意跨平台：PyInstaller 不能跨平台构建。Linux 产物请在 Linux 打包，Windows 产物请在 Windows 打包。

## 快速打包命令
- Linux（在项目根目录）：
  - `python -m PyInstaller --clean --onefile --name OutputPrediction-linux main.py`
- Windows（在项目根目录）：
  - `python -m PyInstaller --clean --onefile --name OutputPrediction-windows main.py`

打包完成后，产物位于 `dist/` 目录。

## 使用构建脚本（推荐）
已在项目根目录新增 `build_pyinstaller.py`，支持清理与平台选择。

- 清理后构建 Linux：
  - `python build_pyinstaller.py --target linux --onefile --clean`
- 清理后构建 Windows：
  - `python build_pyinstaller.py --target windows --onefile --clean`
- 自定义入口：
  - `python build_pyinstaller.py --target linux --entry main.py --onefile`

脚本默认排除 `tests/` 与 `examples/` 模块，避免将无关测试与示例打入可执行文件。

## 使用 spec 文件（高级控制）
当需要精确控制收集的模块、数据或隐藏导入时，可使用项目提供的 spec 文件：

- Linux：
  - `python -m PyInstaller pyinstaller-linux.spec`
- Windows：
  - `python -m PyInstaller pyinstaller-windows.spec`

spec 文件要点：
- 入口为 `main.py`
- 排除 `tests` 与 `examples`
- Windows 版默认添加了 `hiddenimports=['numpy']`

## 常见问题与排除项
- 体积过大：
  - 可添加 `--exclude-module` 排除未使用模块，例如：`--exclude-module matplotlib`
  - 对于 `torch`，建议在 Windows 下使用 `--collect-all torch`，Linux 下保持默认即可。
- 运行失败/缺依赖：
  - 在 Windows 下尝试：`--collect-all torch --collect-submodules numpy`
  - 检查入口脚本是否仅导入必要模块。
- 不需要的目录：
  - 测试与绘图示例通常无需打包，已默认排除 `tests/` 与 `examples/`。
  - 如需排除更多目录，请在命令中添加 `--exclude-module` 或在 spec 的 `excludes` 中补充。

## 清理说明
打包前后如需清理产物与临时文件：
- 使用构建脚本清理：`python build_pyinstaller.py --clean`
- 手动删除：`rm -rf build dist *.spec`

## 复现示例
```bash
# Linux 一次性打包（单文件）
python build_pyinstaller.py --target linux --onefile --clean

# Windows 一次性打包（单文件）
python build_pyinstaller.py --target windows --onefile --clean
```

## 提示
- 如需将入口从 `main.py` 更换为其他脚本，请保持其具备清晰的 CLI 接口。
- 若你只需要训练或测试某个实验，请在运行时传入对应参数，例如：
  - `./OutputPrediction-linux --experiment exp1 --cipher present --rounds 4 --mode train`

以上即为标准打包流程与使用方法，如需更细化的排除列表或对 `torch` 的平台特定优化，我可以继续完善 spec 文件与构建脚本。