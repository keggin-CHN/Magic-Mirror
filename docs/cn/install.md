# 安装指南

## 环境要求

- Node.js 18+
- pnpm 8+
- Python 3.10+
- Rust 与 Tauri 依赖环境（桌面端构建需要）
- Android Studio（Android 端构建需要）

## 前端

```bash
pnpm install
pnpm dev
```

## Python 服务端

```bash
cd src-python
pip install -r requirements.txt
python web_server.py
```

源码本地运行 Web 服务时，如果配置文件不存在，会使用开发默认密码 `123456` 创建本地配置。生产 Web 安装脚本不会使用该默认密码，而是生成一次性随机初始密码。

Windows GPU 环境需要回到仓库根目录，再明确选择一个 ONNX Runtime：

```powershell
# 通用兼容方案：NVIDIA / AMD / Intel
.\scripts\install-windows-ort.ps1 -Runtime directml

# NVIDIA 优化方案，并安装自包含 CUDA/cuDNN 依赖
.\scripts\install-windows-ort.ps1 -Runtime cuda -BundleCudaDependencies
```

不要在同一个 Python 环境中同时安装 `onnxruntime-directml` 和 `onnxruntime-gpu`，它们会覆盖同一个 `onnxruntime` 模块。

## Linux Web 生产安装

使用官方 Web 安装包中的安装脚本：

```bash
sudo ./install-web.sh
```

安装脚本会：

- 生成随机 `VIDEO_TASK_CONFIG_SECRET`（除非你显式传入）。
- 在首次安装且配置不存在时生成随机初始密码。
- 将密码哈希写入 `data/web/config.json`，不会把明文密码写入 systemd 环境。

## GitHub Actions 构建服务端

在仓库 Actions 页面手动运行 **Build Server**。成功后会得到：

- `server_windows_x86_64.zip`：默认 CPU + DirectML 通用包，供桌面端自动下载。
- `server_windows_x86_64_directml.zip`：明确命名的 DirectML 通用包。
- `server_windows_x86_64_cuda.zip`：CPU + CUDA 自包含包，可从 Action Artifacts 下载。

## 桌面端运行

```bash
pnpm tauri dev
```

## Android 端运行

使用 Android Studio 打开 `android-app` 目录，等待 Gradle 同步完成后运行。

## 下载说明

应用内服务端下载地址指向：

https://github.com/keggin-CHN/Magic-Mirror

模型下载地址保持不变。
