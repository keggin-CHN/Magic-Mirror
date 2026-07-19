# API 文档

本文档说明 Magic Mirror 当前 Python 服务端提供的主要 HTTP 接口。

## 基本信息

- 桌面后端默认地址：`http://127.0.0.1:8023`
- Web 后端默认地址：`http://127.0.0.1:8033`
- 桌面后端端口由 `MIRROR_PORT` 配置，Web 后端端口由 `WEB_PORT` 配置。
- Web API 使用 JSON 和 `multipart/form-data`。
- Web API 登录后访问，支持 `Authorization: Bearer <token>`、HttpOnly Cookie、`X-Token`，并临时兼容旧的 `?token=` 查询参数。

## 桌面后端接口

桌面后端入口为 `src-python/server.py`，FastAPI 应用定义在 `src-python/magic/app.py`。

| 路由 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 健康检查 |
| `/prepare` | POST | 预加载模型 |
| `/task/detect-faces` | POST | 检测图片人脸框 |
| `/task/video/detect-faces` | POST | 检测视频关键帧人脸框 |
| `/task/video/gpu-modes` | GET | 查询可用 GPU 模式 |
| `/task` | POST | 创建图片换脸任务 |
| `/task/video` | POST | 创建视频换脸任务 |
| `/task/video/progress/{task_id}` | GET | 查询视频任务进度 |
| `/task/video/ws/{task_id}` | WebSocket | 订阅视频任务进度 |
| `/task/{task_id}` | DELETE | 取消任务 |
| `/file/{file_id}` | GET | 获取结果或上传文件 |
| `/download/{file_id}` | GET / HEAD | 下载结果文件或下载前探测 |

## Web 后端接口

Web 后端入口为 `src-python/web_server.py`，所有 API 路由使用 `/api` 前缀。

| 路由 | 方法 | 说明 |
| --- | --- | --- |
| `/api/status` | GET | 健康检查 |
| `/api/login` | POST | 登录并返回 token，同时设置 HttpOnly Cookie |
| `/api/credential` | POST | 修改登录密码并签发新 token |
| `/api/prepare` | POST | 预加载模型 |
| `/api/upload` | POST | 上传图片或视频 |
| `/api/library` | GET | 列出人脸素材库 |
| `/api/library/upload` | POST | 上传人脸素材 |
| `/api/library/{file_name}` | GET / DELETE | 获取或删除素材库文件 |
| `/api/task/detect-faces` | POST | 检测图片人脸框 |
| `/api/task/video/detect-faces` | POST | 检测视频关键帧人脸框 |
| `/api/task/video/gpu-modes` | GET | 查询可用 GPU 模式 |
| `/api/task` | POST | 创建图片换脸任务 |
| `/api/task/video` | POST | 创建视频换脸任务 |
| `/api/task/video/progress/{task_id}` | GET | 查询视频任务进度 |
| `/api/task/video/ws/{task_id}` | WebSocket | 订阅视频任务进度 |
| `/api/task/{task_id}` | DELETE | 取消任务 |
| `/api/file/{file_id}` | GET | 获取结果或上传文件 |
| `/api/download/{file_id}` | GET / HEAD | 下载结果文件或下载前探测 |

## 常见响应

登录成功：

```json
{
  "token": "..."
}
```

上传成功：

```json
{
  "fileId": "...",
  "url": "/api/file/...",
  "type": "image",
  "name": "input.jpg"
}
```

任务提交成功：

```json
{
  "task_id": "...",
  "status": "queued"
}
```

错误响应通常包含 `error` 字段：

```json
{
  "error": "missing-params"
}
```

常见错误码包括 `unauthorized`、`missing-params`、`file-not-found`、`unsupported-image-format`、`unsupported-video-format`、`file-too-large`、`task-already-running`、`cancelled` 和 `internal`。

## 部署初始化入口

Web 后端支持一次性初始化配置：

```bash
WEB_INITIAL_PASSWORD="change-me" python web_server.py --init-config
```

该命令只在配置不存在时写入密码哈希，不启动 HTTP 服务。生产 Linux 安装脚本会自动生成随机初始密码并调用该入口。
