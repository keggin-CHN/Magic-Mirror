# Magic Mirror

> 一面写在像素里的魔镜：让影像轻轻流转，让面容在光里重生。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Web](https://img.shields.io/github/actions/workflow/status/keggin-CHN/Magic-Mirror/build-web.yaml?label=Build%20Web)](https://github.com/keggin-CHN/Magic-Mirror/actions/workflows/build-web.yaml)
[![Build Server](https://img.shields.io/github/actions/workflow/status/keggin-CHN/Magic-Mirror/build-server.yaml?label=Build%20Server)](https://github.com/keggin-CHN/Magic-Mirror/actions/workflows/build-server.yaml)

---

## 简介

Magic Mirror 是一款跨平台 AI 换脸应用，支持图片与视频的一键换脸。覆盖桌面端（Windows / macOS / Linux）、Web 端与 Android 端，所有处理均在本地完成，不上传任何图片到第三方服务。

---

## 功能特性

- **图片换脸** — 单脸、多脸、手动选区精准换脸
- **视频换脸** — 基于 MediaCodec 逐帧解码与重编码，支持音轨保留
- **多人脸源绑定** — 一次任务中为不同区域绑定不同人脸源
- **GPU 加速** — 桌面端支持 CUDA / DirectML，自动检测可用加速
- **Web 模式** — 浏览器直接访问后端 API，无需安装客户端
- **多语言** — 内置中文、英文等多语言切换（i18next）
- **隐私优先** — 所有处理本地完成，零上传

---

## 快速开始

### 环境要求

| 组件 | 最低版本 | 用途 |
|------|---------|------|
| Node.js | ≥ 18 | 前端构建 |
| pnpm | ≥ 8 | 包管理 |
| Python | ≥ 3.10 | 后端推理服务 |
| Rust | ≥ 1.70 | Tauri 桌面壳（仅桌面端） |
| Android Studio + JDK 17 | — | Android 端构建 |

### 前端开发

```bash
pnpm install
pnpm dev          # 开发服务器，默认端口 5173
pnpm build        # 生产构建
```

### Python 后端

```bash
cd src-python
pip install -r requirements.txt

# 桌面后端（端口 8023）
python server.py

# Web 后端（端口 8033，带 JWT 鉴权）
python web_server.py

# 自定义端口
MIRROR_HOST=0.0.0.0 MIRROR_PORT=9000 python server.py
```

Windows 下 `tinyface` 会先拉取 CPU 版 ONNX Runtime。需要 GPU 开发环境时，
请在仓库根目录执行以下二选一命令，确保 DirectML 和 CUDA 不会互相覆盖：

```powershell
# 推荐：兼容 NVIDIA / AMD / Intel，无需 CUDA Toolkit
.\scripts\install-windows-ort.ps1 -Runtime directml

# NVIDIA CUDA；附加开关会安装并打包 CUDA 12.9 / cuDNN 9 运行库
.\scripts\install-windows-ort.ps1 -Runtime cuda -BundleCudaDependencies
```

### Windows 服务端发布包

`Build Server` GitHub Action 会分别构建并验证两个互斥的 ONNX Runtime：

Windows Runner 使用 Python 3.11：DirectML 固定为 ORT 1.23.0，CUDA 固定为
ORT 1.27.0（RTX 50 系列需要 1.27+ 才能实际执行 CUDA kernel）。

- `server_windows_x86_64.zip`：默认通用包，CPU + DirectML，保持客户端现有下载地址兼容。
- `server_windows_x86_64_directml.zip`：与默认包内容相同，名称明确的 DirectML 包。
- `server_windows_x86_64_cuda.zip`：CPU + CUDA，自带 CUDA 12.9、cuDNN、cuBLAS 和 cuFFT DLL，保留在 Actions Artifact 中。

由于 CUDA 自包含包包含模型和完整 GPU 运行库，可能超过 GitHub Release 的 2 GiB
单文件限制；稳定 Release 只上传 DirectML 包，CUDA 包可从对应 Action 的 Artifacts
下载。

官方 `onnxruntime-directml` 与 `onnxruntime-gpu` 都提供同名 Python 模块，不能
安全安装在同一 Python 进程中。因此发布流程使用两个独立运行时产物，而不是把
两个 wheel 覆盖到同一个目录。每个产物都会运行冻结后的 `server.exe` 检查目标
Provider；检查失败时 Action 会直接失败，不再发布伪 GPU 包。

### Docker

```bash
docker build -t magic-mirror:latest .
docker run -p 8023:8023 magic-mirror:latest            # 桌面后端
docker run -p 8033:8033 -e WEB_PORT=8033 magic-mirror:latest python web_server.py  # Web 模式
```

### Makefile

```bash
make dev            # 启动桌面后端
make web            # 启动 Web 后端
make build          # 前端构建
make docker-build   # 构建 Docker 镜像
```

### Tauri 桌面端

```bash
pnpm tauri dev      # 开发模式
pnpm tauri build    # 打包
```

### Android 端

```bash
cd android-app
./gradlew assembleDebug
# 产物: android-app/app/build/outputs/apk/debug/app-debug.apk
```

---

## 项目结构

```
Magic-Mirror/
├── src/                    # 前端 React + TypeScript（Vite + UnoCSS）
│   ├── pages/              # 页面组件（Mirror、Login、Launch 等）
│   ├── hooks/              # 自定义 Hooks（useSwapFace 等）
│   ├── services/           # API 客户端与工具函数
│   └── assets/locales/     # 多语言资源（zh / en）
├── src-python/             # Python 后端（FastAPI + ONNX Runtime）
│   ├── magic/              # 换脸核心模块（face detection、swap engine）
│   └── web_server.py       # Web 模式 HTTP 服务（端口 8033，JWT 鉴权）
├── src-tauri/              # Tauri 2.0 Rust 桌面壳
├── android-app/            # Android 原生应用（Java + ONNX Runtime）
├── docs/                   # 项目文档（中英双语 + API 参考）
├── scripts/                # 构建脚本
└── .github/workflows/      # CI/CD（多平台构建）
```

---

## 模型文件

推理所需模型需放在 `src-python/models/` 目录下：

| 文件名 | 用途 |
|--------|------|
| `scrfd_2.5g.onnx` | 人脸检测 |
| `arcface_w600k_r50.onnx` | 人脸特征提取 |
| `inswapper_128_fp16.onnx` | 人脸替换 |
| `gfpgan_1.4.onnx` | 人脸增强 |

模型下载方式见 [安装指南](docs/cn/install.md)。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MIRROR_HOST` | `0.0.0.0` | 监听地址 |
| `MIRROR_PORT` | `8023` | 桌面后端端口 |
| `WEB_HOST` | `0.0.0.0` | Web 后端监听地址 |
| `WEB_PORT` | `8033` | Web 后端端口 |
| `WEB_DATA_DIR` | `data/web` | Web 上传、素材库和配置目录 |
| `WEB_DIST_DIR` | `dist-web` | Web 前端静态资源目录 |
| `WEB_INITIAL_PASSWORD` | 无 | `web_server.py --init-config` 首次初始化密码 |
| `VIDEO_TASK_CONFIG_SECRET` | 开发默认值 | 生产环境应使用随机值 |

---

## Web 模式安全特性

`src-python/web_server.py` 内置生产级安全加固：

- **Token / Cookie 鉴权** — 登录获取 token，同时设置 HttpOnly Cookie；API 支持 `Authorization: Bearer <token>`、Cookie 和兼容 token 入口
- **TTL 垃圾回收** — 上传文件 24h、结果 4h、进度 6h 自动清理
- **路径穿越防护** — `os.path.commonpath` 校验所有文件访问
- **文件名清洗** — 正则过滤危险字符
- **上传大小限制** — 图片 50MB / 视频 2GB
- **节流 GC** — `before_request` 钩子按需触发清理

---

## API 概览（Web 模式）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/login` | POST | 登录获取 token |
| `/api/upload` | POST | 上传图片 / 视频 |
| `/api/library` | GET / POST | 人脸库管理 |
| `/api/task/detect-faces` | POST | 检测图片人脸框 |
| `/api/task/video/detect-faces` | POST | 检测视频关键帧人脸框 |
| `/api/task` | POST | 创建图片换脸任务 |
| `/api/task/video` | POST | 创建视频换脸任务 |
| `/api/task/video/progress/<id>` | GET | 查询视频任务进度 |
| `/api/task/<id>` | DELETE | 取消任务 |
| `/api/file/<id>` | GET | 获取结果文件 |
| `/api/download/<id>` | GET / HEAD | 下载结果文件；HEAD 用于下载前探测 |

完整字段定义、请求/响应示例与错误码见 [中文 API 文档](docs/cn/API.md) / [English API Reference](docs/en/API.md)。

---

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 UI | React 18、TypeScript、UnoCSS、Vite |
| 状态管理 | xsta |
| 多语言 | i18next + react-i18next |
| 桌面壳 | Tauri 2.0（Rust） |
| 后端 | Python 3.10、FastAPI、ONNX Runtime |
| 模型 | SCRFD + ArcFace + InSwapper + GFPGAN |
| Android | Java、ONNX Runtime Android、MediaCodec |
| CI/CD | GitHub Actions（多平台构建） |

---

## 性能优化

- **桌面端** — GPU（CUDA / DirectML）加速，CPU 模式最多 6 worker，GPU 模式 2 worker
- **视频处理** — 分段并行 + 关键帧追踪，多人换脸单 worker 顺序提交保证时序一致
- **Android** — YUV ↔ RGB 使用定点整数运算 + bulk-copy，相对浮点逐像素实现快 5-15x
- **前端轮询** — 自适应间隔（500ms - 2s）+ 指数退避，连续失败 8 次自动放弃
- **网络层** — 所有 fetch 包装超时（默认 30s / 上传 10min / 进度 15s）

---

## 文档

- [中文文档](docs/cn/readme.md) | [English Docs](docs/en/readme.md)
- [安装指南](docs/cn/install.md) | [Install Guide](docs/en/install.md)
- [使用说明](docs/cn/usage.md) | [Usage Guide](docs/en/usage.md)
- [常见问题](docs/cn/faq.md) | [FAQ](docs/en/faq.md)
- [API 参考](docs/cn/API.md) | [API Reference](docs/en/API.md)

---

## 贡献

欢迎提交 Issue 和 Pull Request！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 许可证

[MIT License](LICENSE)

---

## 作者

- **keggin** — [GitHub](https://github.com/keggin-CHN) | zhou239829001@gmail.com
