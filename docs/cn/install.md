# 安装指南

## 环境要求

- Node.js 18+
- Python 3.10+
- Rust 与 Tauri 依赖环境（桌面端构建需要）
- Android Studio（Android 端构建需要）

## 前端安装

```bash
npm install
npm run dev
```

## Python 服务端安装

```bash
cd src-python
pip install -r requirements.txt
python web_server.py
```

Windows GPU 环境需要回到仓库根目录，再明确选择一个 ONNX Runtime：

```powershell
# 通用兼容方案：NVIDIA / AMD / Intel
.\scripts\install-windows-ort.ps1 -Runtime directml

# NVIDIA 优化方案，并安装自包含 CUDA/cuDNN 依赖
.\scripts\install-windows-ort.ps1 -Runtime cuda -BundleCudaDependencies
```

不要在同一个 Python 环境中同时安装 `onnxruntime-directml` 和
`onnxruntime-gpu`，它们会覆盖同一个 `onnxruntime` 模块。

## GitHub Actions 构建服务端

在仓库的 Actions 页面手动运行 **Build Server**。成功后会得到：

构建使用 Python 3.11；DirectML 使用 ORT 1.23.0，CUDA 使用 ORT 1.27.0，
以覆盖包括 RTX 50 系列在内的新 NVIDIA 架构。

- `server_windows_x86_64.zip`：默认 CPU + DirectML 通用包，供现有桌面端自动下载。
- `server_windows_x86_64_directml.zip`：明确命名的 DirectML 通用包。
- `server_windows_x86_64_cuda.zip`：CPU + CUDA 自包含包（CUDA 12.9 / cuDNN 9），从 Action 的 Artifacts 下载。

CUDA 包同时包含模型和完整运行库，可能超过 GitHub Release 的 2 GiB 单文件限制，
所以稳定 Release 只上传 DirectML 包；这不会影响 Action 生成 CUDA 包。

默认推荐 DirectML 包，因为它不要求 CUDA Toolkit，并兼容 NVIDIA、AMD 和 Intel
显卡。CUDA 包更大，适合 NVIDIA 显卡和追求 CUDA 性能的用户。

## 桌面端运行

```bash
npm run tauri dev
```

## Android 端运行

使用 Android Studio 打开 `android-app` 目录，等待 Gradle 同步完成后运行。

## 下载说明

应用内服务端下载地址指向：https://github.com/keggin-CHN/Magic-Mirror

模型下载地址保持不变。
