# API 文档 / API Reference

本文档说明 Magic Mirror Python 服务端提供的主要 HTTP 接口。默认服务地址通常为 `http://127.0.0.1:8000`，实际端口以运行配置为准。

## 基本信息

- 数据格式：`multipart/form-data` 或 `application/json`
- 返回格式：`application/json`
- 文件字段通常使用图片或视频文件上传
- 模型下载地址保持不变
- 项目仓库：https://github.com/keggin-CHN/Magic-Mirror

## 健康检查

### `GET /`

用于检查服务是否启动。

**响应示例：**

```json
{
  "message": "Magic Mirror server is running"
}
```

## 单图换脸

### `POST /swap`

上传源人脸图片与目标图片，返回换脸结果。

**请求字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `source` | file | 是 | 源人脸图片 |
| `target` | file | 是 | 目标图片 |
| `regions` | string | 否 | 选区 JSON 字符串 |

**响应示例：**

```json
{
  "success": true,
  "output": "/path/to/output.jpg"
}
```

## 视频换脸

### `POST /swap-video`

上传源人脸图片与目标视频，创建视频换脸任务。

**请求字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `source` | file | 是 | 源人脸图片 |
| `target` | file | 是 | 目标视频 |
| `regions` | string | 否 | 选区 JSON 字符串 |
| `acceleration` | string | 否 | 加速模式，如 `cpu`、`directml`、`cuda` |

**响应示例：**

```json
{
  "success": true,
  "taskId": "task-id"
}
```

## 查询任务进度

### `GET /task/{taskId}`

查询视频换脸任务状态。

**响应示例：**

```json
{
  "success": true,
  "status": "processing",
  "progress": 50,
  "stage": "Processing video frames",
  "eta": "00:01:30"
}
```

常见状态：

- `queued`
- `processing`
- `completed`
- `failed`
- `cancelled`

## 下载结果

### `GET /download/{fileName}`

下载已生成的图片或视频文件。

## 任务配置

### `POST /task-config`

生成可复用的任务配置 ID，用于服务端执行或复用相同配置。

**响应示例：**

```json
{
  "success": true,
  "configId": "config-id"
}
```

## 错误响应

接口失败时通常返回：

```json
{
  "success": false,
  "error": "error message"
}
```

常见错误原因：

- 未检测到人脸
- 文件格式不支持
- 上传文件缺失
- 视频打开失败
- 输出文件写入失败
- ffmpeg 音频合成失败

## English Summary

Magic Mirror exposes HTTP endpoints for image face swap, video face swap, task progress polling, result download, and task configuration generation. The default local server is usually `http://127.0.0.1:8000`.

Repository: https://github.com/keggin-CHN/Magic-Mirror