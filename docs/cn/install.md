# 安装教程

![](../assets/banner.jpg)

## 下载地址

首先根据你的操作系统，下载对应的 APP 安装包并安装：

1. Windows: [MagicMirror_1.0.0_windows_x86_64.exe](https://gh-proxy.com/github.com/idootop/MagicMirror/releases/download/app-v1.0.0/MagicMirror_1.0.0_windows_x86_64.exe)
2. macOS: [MagicMirror_1.0.0_macos_universal.dmg](https://gh-proxy.com/github.com/idootop/MagicMirror/releases/download/app-v1.0.0/MagicMirror_1.0.0_macos_universal.dmg)

## 安装依赖（Windows）

![](../assets/msvc.png)

对于 Windows 用户，需要先安装 Microsoft Visual C++ 运行时才能正常运行。

下载地址：https://aka.ms/vs/17/release/vc_redist.x64.exe

## 下载模型

首次启动 APP，需要下载模型文件（默认会自动下载），初始化成功后才能使用。

![](../assets/download.png)

如果你的下载进度一直是 0，或者下到一半卡住了，请按照下面的步骤手动初始化：

**下载模型文件**

首先根据你的操作系统，选择对应的模型文件：
https://gh-proxy.com/github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_windows_x86_64.zip

- [server_windows_x86_64.zip](https://gh-proxy.com/github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_windows_x86_64.zip)
- [server_macos_x86_64.zip](https://gh-proxy.com/github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_macos_x86_64.zip)

> 如果你访问不了上面的 GitHub 下载链接，可以使用国内的[下载地址](https://pan.quark.cn/s/b8ad043794bb)

**解压下载的文件**

解压后应该是一个文件夹，把它重命名成: `MagicMirror` 然后移动到你电脑的 `HOME` 目录下，比如：

![](../assets/windows-home.png)

![](../assets/macos-home.png)

完成上面两步后，重启 MagicMirror 应该就能正常启动了。

## 启动 APP

下载完模型文件后，第一次启动应用可能比较缓慢，耐心等待即可。

![](../assets/launch.png)

> 一般 3 分钟以内即可启动成功，如果超过 10 分钟还未正常启动，请查看[常见问题](./faq.md)

## Linux 无桌面服务器部署（Ubuntu/Debian/CentOS/RHEL）

针对超算/服务器场景（无桌面终端），建议部署 Web 服务端并远程提交任务。

### 1）准备服务端发布包

确保服务器目录内包含：

- `web_server.dist/`
- `dist-web/`
- 可选发布压缩包：`magicmirror_web_*.tar.gz` 或 `web_linux_x86_64.zip`

### 2）运行跨发行版安装脚本

使用统一 Linux 安装脚本：

```bash
sudo INSTALL_DIR=/opt/magicmirror \
  WEB_HOST=0.0.0.0 \
  WEB_PORT=21859 \
  WEB_UI_PORT=15129 \
  SERVICE_NAME=magic-mirror-web \
  SERVICE_USER=root \
  VIDEO_TASK_CONFIG_SECRET='请替换为强密钥' \
  bash ./scripts/install-server-linux.sh
```

脚本路径：`scripts/install-server-linux.sh`

脚本能力：

- 自动识别包管理器（`apt` / `dnf` / `yum`）
- 安装运行依赖（`ffmpeg`、`nginx`、OpenCV 相关运行库等）
- 优先配置 `systemd` 服务
- 老系统无 `systemd` 时自动 fallback 到 `nohup` 常驻
- 自动配置反向代理：`UI -> /api -> WEB_PORT`

### 3）服务管理（systemd）

如果系统支持 systemd：

```bash
sudo systemctl status magic-mirror-web --no-pager
sudo systemctl restart magic-mirror-web
sudo journalctl -u magic-mirror-web -f
```

### 4）端口与防火墙

请在云安全组和主机防火墙放行：

- `WEB_UI_PORT`（默认 `15129`）：浏览器访问 UI
- `WEB_PORT`（默认 `21859`）：后端 API（通常被 UI 反向代理）

### 5）“仅查看任务ID”工作流（config-only）

当你在页面勾选 **仅查看任务ID（不在本机执行）** 后：

- 服务端只生成任务配置并返回 `configId`
- 不会在当前机器执行换脸
- 后续可携带该 `configId` 再发起执行请求（由服务器跑任务）

当前实现已支持带签名且带 TTL 的配置 token，不再只依赖进程内内存态 ID。
生产环境务必设置稳定且强度足够的 `VIDEO_TASK_CONFIG_SECRET`。

### 6）老系统兼容建议

- 默认仓库没有 `ffmpeg` 时，请先启用发行版扩展仓库再安装
- 不想使用 nginx 时可设置 `SKIP_NGINX=1`，直接暴露 API 端口
- 无 `systemd` 时，脚本会写入 `data/web/` 下的 pid/log 并后台运行

## 遇到问题？

大部分问题都能在「[常见问题](./faq.md)」里找到答案，如果你还有其他问题，请在此[提交反馈](https://github.com/idootop/MagicMirror/issues)。
