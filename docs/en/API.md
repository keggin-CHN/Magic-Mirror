# API Reference

This document describes the current HTTP APIs exposed by the Magic Mirror Python servers.

## Basics

- Desktop server default URL: `http://127.0.0.1:8023`
- Web server default URL: `http://127.0.0.1:8033`
- Desktop server port is configured with `MIRROR_PORT`; Web server port is configured with `WEB_PORT`.
- Web APIs use JSON and `multipart/form-data`.
- Web APIs require authentication after login. Supported auth sources are `Authorization: Bearer <token>`, HttpOnly Cookie, `X-Token`, and the legacy `?token=` query parameter.

## Desktop Server APIs

The desktop server entrypoint is `src-python/server.py`; the FastAPI app is defined in `src-python/magic/app.py`.

| Route | Method | Description |
| --- | --- | --- |
| `/` | GET | Health check |
| `/prepare` | POST | Preload models |
| `/task/detect-faces` | POST | Detect image face boxes |
| `/task/video/detect-faces` | POST | Detect face boxes on a video key frame |
| `/task/video/gpu-modes` | GET | List available GPU modes |
| `/task` | POST | Create an image face swap task |
| `/task/video` | POST | Create a video face swap task |
| `/task/video/progress/{task_id}` | GET | Get video task progress |
| `/task/video/ws/{task_id}` | WebSocket | Subscribe to video task progress |
| `/task/{task_id}` | DELETE | Cancel a task |
| `/file/{file_id}` | GET | Fetch a result or uploaded file |
| `/download/{file_id}` | GET / HEAD | Download a result file or probe before download |

## Web Server APIs

The Web server entrypoint is `src-python/web_server.py`; all API routes use the `/api` prefix.

| Route | Method | Description |
| --- | --- | --- |
| `/api/status` | GET | Health check |
| `/api/login` | POST | Log in, return a token, and set an HttpOnly Cookie |
| `/api/credential` | POST | Change the login password and issue a new token |
| `/api/prepare` | POST | Preload models |
| `/api/upload` | POST | Upload an image or video |
| `/api/library` | GET | List face library items |
| `/api/library/upload` | POST | Upload a face library item |
| `/api/library/{file_name}` | GET / DELETE | Fetch or delete a library file |
| `/api/task/detect-faces` | POST | Detect image face boxes |
| `/api/task/video/detect-faces` | POST | Detect face boxes on a video key frame |
| `/api/task/video/gpu-modes` | GET | List available GPU modes |
| `/api/task` | POST | Create an image face swap task |
| `/api/task/video` | POST | Create a video face swap task |
| `/api/task/video/progress/{task_id}` | GET | Get video task progress |
| `/api/task/video/ws/{task_id}` | WebSocket | Subscribe to video task progress |
| `/api/task/{task_id}` | DELETE | Cancel a task |
| `/api/file/{file_id}` | GET | Fetch a result or uploaded file |
| `/api/download/{file_id}` | GET / HEAD | Download a result file or probe before download |

## Common Responses

Successful login:

```json
{
  "token": "..."
}
```

Successful upload:

```json
{
  "fileId": "...",
  "url": "/api/file/...",
  "type": "image",
  "name": "input.jpg"
}
```

Task submission:

```json
{
  "task_id": "...",
  "status": "queued"
}
```

Error responses usually include an `error` field:

```json
{
  "error": "missing-params"
}
```

Common error codes include `unauthorized`, `missing-params`, `file-not-found`, `unsupported-image-format`, `unsupported-video-format`, `file-too-large`, `task-already-running`, `cancelled`, and `internal`.

## Deployment Initialization

The Web server supports one-time config initialization:

```bash
WEB_INITIAL_PASSWORD="change-me" python web_server.py --init-config
```

This command writes the password hash only when the config does not exist, and it does not start the HTTP server. The production Linux installer generates a random initial password and calls this entrypoint automatically.
