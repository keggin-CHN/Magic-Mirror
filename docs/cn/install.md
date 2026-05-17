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

## 桌面端运行

```bash
npm run tauri dev
```

## Android 端运行

使用 Android Studio 打开 `android-app` 目录，等待 Gradle 同步完成后运行。

## 下载说明

应用内服务端下载地址指向：https://github.com/keggin-CHN/Magic-Mirror

模型下载地址保持不变。
