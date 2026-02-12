import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from async_tasks import AsyncTask
from bottle import Bottle, request, response, static_file

from magic.face import (
    detect_face_boxes_in_image,
    detect_face_boxes_in_video,
    load_models,
    swap_face,
    swap_face_regions,
    swap_face_regions_by_sources,
    swap_face_video,
    swap_face_video_by_sources,
)

app = Bottle()

# https://github.com/bottlepy/bottle/issues/881#issuecomment-244024649
app.plugins[0].json_dumps = lambda *args, **kwargs: json.dumps(
    *args, ensure_ascii=False, **kwargs
).encode("utf8")

ALLOWED_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

ALLOWED_VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WEB_DATA_DIR = os.path.join(BASE_DIR, "data", "web")
UPLOADS_DIR = os.path.join(WEB_DATA_DIR, "uploads")
LIBRARY_DIR = os.path.join(WEB_DATA_DIR, "library")
CONFIG_PATH = os.path.join(WEB_DATA_DIR, "config.json")
DIST_DIR = os.path.join(BASE_DIR, "dist-web")

TOKEN_TTL_SECONDS = 7 * 24 * 3600

TOKENS: Dict[str, float] = {}
TOKENS_LOCK = threading.RLock()

UPLOADS: Dict[str, Dict[str, str]] = {}
UPLOADS_LOCK = threading.RLock()

RESULTS: Dict[str, Dict[str, object]] = {}
RESULTS_LOCK = threading.RLock()

VIDEO_TASK_PROGRESS = {}
VIDEO_TASK_PROGRESS_LOCK = threading.RLock()


def _ensure_dirs():
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(LIBRARY_DIR, exist_ok=True)


_ensure_dirs()


def _load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        _save_config({"password": "123456"})
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _issue_token() -> str:
    token = uuid.uuid4().hex
    with TOKENS_LOCK:
        TOKENS[token] = time.time()
    return token


def _cleanup_tokens() -> None:
    now = time.time()
    with TOKENS_LOCK:
        expired = [
            token
            for token, created in TOKENS.items()
            if now - created > TOKEN_TTL_SECONDS
        ]
        for token in expired:
            TOKENS.pop(token, None)


def _extract_token() -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Token") or request.query.get("token")


def _require_auth() -> bool:
    _cleanup_tokens()
    token = _extract_token()
    if not token:
        response.status = 401
        return False
    with TOKENS_LOCK:
        if token not in TOKENS:
            response.status = 401
            return False
    return True


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _validate_file(path: str, allowed_exts: set, *, missing_code: str):
    if not path:
        raise RuntimeError("missing-params")
    if not os.path.exists(path):
        raise FileNotFoundError("file-not-found")
    if _ext(path) not in allowed_exts:
        raise RuntimeError(missing_code)


def _simplify_task_error(err: object) -> str:
    msg = (str(err) if err is not None else "").lower()
    codes = [
        "missing-params",
        "missing-face-sources",
        "invalid-face-source-binding",
        "face-source-not-found",
        "file-not-found",
        "unsupported-image-format",
        "unsupported-video-format",
        "image-decode-failed",
        "no-face-detected",
        "no-face-in-selected-regions",
        "swap-failed",
        "video-open-failed",
        "video-write-failed",
        "video-output-missing",
        "video-frame-read-failed",
        "output-write-failed",
    ]
    for code in codes:
        if code in msg:
            return code
    return "internal"


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "upload")
    return base.replace(" ", "_")


def _save_upload(upload_file, dest_dir: str):
    filename = _sanitize_filename(upload_file.filename)
    ext = os.path.splitext(filename)[1].lower()
    file_id = uuid.uuid4().hex
    safe_name = f"{file_id}{ext}" if ext else file_id
    save_path = os.path.join(dest_dir, safe_name)
    upload_file.save(save_path, overwrite=True)
    return file_id, save_path, safe_name


def _register_upload(file_id: str, path: str, kind: str) -> None:
    with UPLOADS_LOCK:
        UPLOADS[file_id] = {"path": path, "kind": kind}


def _register_result(result_path: str, delete_paths: List[str]) -> str:
    result_id = uuid.uuid4().hex
    with RESULTS_LOCK:
        RESULTS[result_id] = {
            "path": result_path,
            "delete_paths": delete_paths,
            "name": os.path.basename(result_path),
        }
    return result_id


def _get_upload_path(file_id: str) -> Optional[str]:
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        return item.get("path") if item else None


def _get_upload_kind(file_id: str) -> Optional[str]:
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        return item.get("kind") if item else None


def _get_result_info(file_id: str) -> Optional[Dict[str, object]]:
    with RESULTS_LOCK:
        info = RESULTS.get(file_id)
        return info.copy() if info else None


def _remove_upload_by_path(path: str) -> None:
    with UPLOADS_LOCK:
        to_remove = [key for key, item in UPLOADS.items() if item.get("path") == path]
        for key in to_remove:
            UPLOADS.pop(key, None)


def _safe_delete(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _list_library_items() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not os.path.isdir(LIBRARY_DIR):
        return items
    for name in sorted(os.listdir(LIBRARY_DIR)):
        path = os.path.join(LIBRARY_DIR, name)
        if not os.path.isfile(path):
            continue
        if _ext(path) not in ALLOWED_IMAGE_EXTS:
            continue
        items.append(
            {
                "id": name,
                "name": name,
                "url": f"/api/library/{name}",
            }
        )
    return items


def _get_library_path(item_id: str) -> Optional[str]:
    if not item_id:
        return None
    path = os.path.join(LIBRARY_DIR, os.path.basename(item_id))
    if not os.path.exists(path):
        return None
    if _ext(path) not in ALLOWED_IMAGE_EXTS:
        return None
    return path


def _set_video_task_progress(task_id: str, **updates):
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id, {})
        state.update(updates)
        VIDEO_TASK_PROGRESS[task_id] = state


def _get_video_task_progress(task_id: str):
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id)
        if not state:
            return {
                "status": "idle",
                "progress": 0,
                "etaSeconds": None,
                "stage": None,
            }
        return state.copy()


def _cleanup_result(result_id: str, delete_paths: List[str]) -> None:
    with RESULTS_LOCK:
        RESULTS.pop(result_id, None)
    for path in delete_paths:
        _safe_delete(path)
        _remove_upload_by_path(path)


def _stream_and_cleanup(path: str, result_id: str, delete_paths: List[str]):
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    finally:
        _cleanup_result(result_id, delete_paths)


# Enable CORS
@app.hook("after_request")
def enable_cors():
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header(
        "Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS"
    )
    response.set_header(
        "Access-Control-Allow-Headers", "Authorization,Content-Type,X-Token"
    )


@app.route("/api/<path:path>", method=["OPTIONS"])
def handle_options(path):
    response.status = 200
    return {}


@app.get("/api/status")
def status():
    return {"status": "running"}


@app.post("/api/prepare")
def prepare():
    if request.method == "OPTIONS":
        return {}
    return {"success": load_models()}


@app.post("/api/login")
def login():
    if request.method == "OPTIONS":
        return {}
    body = request.json or {}
    password = str(body.get("password", "")).strip()
    cfg = _load_config()
    if password != cfg.get("password"):
        response.status = 401
        return {"error": "invalid-credential"}
    token = _issue_token()
    return {"token": token}


@app.post("/api/credential")
def update_credential():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    body = request.json or {}
    new_password = str(body.get("password", "")).strip()
    if not new_password:
        response.status = 400
        return {"error": "missing-params"}
    _save_config({"password": new_password})
    return {"success": True}


@app.get("/api/library")
def list_library():
    if not _require_auth():
        return {"error": "unauthorized"}
    return {"items": _list_library_items()}


@app.post("/api/library/upload")
def upload_library():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    upload_file = request.files.get("file")
    if not upload_file:
        response.status = 400
        return {"error": "missing-params"}
    ext = _ext(upload_file.filename or "")
    if ext not in ALLOWED_IMAGE_EXTS:
        response.status = 400
        return {"error": "unsupported-image-format"}
    _, save_path, safe_name = _save_upload(upload_file, LIBRARY_DIR)
    return {
        "id": safe_name,
        "name": safe_name,
        "url": f"/api/library/{safe_name}",
    }


@app.get("/api/library/<filename>")
def get_library_file(filename):
    path = os.path.join(LIBRARY_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        response.status = 404
        return {"error": "file-not-found"}
    return static_file(os.path.basename(path), root=os.path.dirname(path))


@app.post("/api/upload")
def upload_file():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    upload_file = request.files.get("file")
    if not upload_file:
        response.status = 400
        return {"error": "missing-params"}
    ext = _ext(upload_file.filename or "")
    if ext in ALLOWED_IMAGE_EXTS:
        kind = "image"
    elif ext in ALLOWED_VIDEO_EXTS:
        kind = "video"
    else:
        response.status = 400
        return {"error": "unsupported-file-format"}
    file_id, save_path, _ = _save_upload(upload_file, UPLOADS_DIR)
    _register_upload(file_id, save_path, kind)
    return {
        "fileId": file_id,
        "url": f"/api/file/{file_id}",
        "type": kind,
        "name": upload_file.filename or "upload",
    }


@app.get("/api/file/<file_id>")
def get_file(file_id):
    path = _get_upload_path(file_id)
    if not path:
        info = _get_result_info(file_id)
        if info:
            path = info.get("path")
    if not path or not os.path.exists(path):
        response.status = 404
        return {"error": "file-not-found"}
    response.set_header("Cache-Control", "no-store")
    return static_file(os.path.basename(path), root=os.path.dirname(path))


@app.get("/api/download/<file_id>")
def download_file(file_id):
    info = _get_result_info(file_id)
    if not info:
        response.status = 404
        return {"error": "file-not-found"}
    path = info.get("path")
    delete_paths = list(info.get("delete_paths") or [])
    if not path or not os.path.exists(path):
        response.status = 404
        return {"error": "file-not-found"}
    filename = info.get("name") or os.path.basename(path)
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    response.content_type = content_type
    response.set_header("Content-Disposition", f'attachment; filename="{filename}"')
    return _stream_and_cleanup(path, file_id, delete_paths)


@app.route("/api/task/detect-faces", method=["POST", "OPTIONS"])
def detect_faces_for_image():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    try:
        body = request.json or {}
        input_id = body.get("inputFileId")
        regions = body.get("regions")
        input_path = _get_upload_path(input_id)
        if not input_path:
            response.status = 400
            return {"error": "file-not-found"}
        _validate_file(
            input_path,
            ALLOWED_IMAGE_EXTS,
            missing_code="unsupported-image-format",
        )
        if regions is not None and not isinstance(regions, list):
            response.status = 400
            return {"error": "missing-params"}
        result = detect_face_boxes_in_image(input_path, regions=regions)
        return {"regions": result}
    except Exception as e:
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.route("/api/task/video/detect-faces", method=["POST", "OPTIONS"])
def detect_faces_for_video():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    try:
        body = request.json or {}
        input_id = body.get("inputFileId")
        key_frame_ms = body.get("keyFrameMs", 0)
        regions = body.get("regions")
        input_path = _get_upload_path(input_id)
        if not input_path:
            response.status = 400
            return {"error": "file-not-found"}
        _validate_file(
            input_path,
            ALLOWED_VIDEO_EXTS,
            missing_code="unsupported-video-format",
        )
        if regions is not None and not isinstance(regions, list):
            response.status = 400
            return {"error": "missing-params"}
        try:
            key_frame_ms = int(float(key_frame_ms or 0))
        except (TypeError, ValueError):
            key_frame_ms = 0
        result = detect_face_boxes_in_video(
            input_path,
            key_frame_ms=max(0, key_frame_ms),
            regions=regions,
        )
        return result
    except Exception as e:
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.route("/api/task", method=["POST", "OPTIONS"])
def create_task():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    try:
        body = request.json or {}
        task_id = body.get("id")
        input_id = body.get("inputFileId")
        regions = body.get("regions")
        target_face_id = body.get("targetFaceId")
        face_sources = body.get("faceSources")
        has_face_sources = "faceSources" in body
        if not all([task_id, input_id]):
            response.status = 400
            return {"error": "missing-params"}
        input_path = _get_upload_path(input_id)
        if not input_path:
            response.status = 400
            return {"error": "file-not-found"}
        try:
            _validate_file(
                input_path,
                ALLOWED_IMAGE_EXTS,
                missing_code="unsupported-image-format",
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {"error": _simplify_task_error(e)}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status = 400
                return {"error": "missing-face-sources"}
            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status = 400
                    return {"error": "missing-face-sources"}
                source_id = source.get("id")
                source_path = _get_library_path(str(source_id))
                if not source_id or not source_path:
                    response.status = 400
                    return {"error": "missing-face-sources"}
                source_map[str(source_id)] = source_path
            if regions:
                if not isinstance(regions, list):
                    response.status = 400
                    return {"error": "invalid-face-source-binding"}
                for region in regions:
                    if not isinstance(region, dict):
                        response.status = 400
                        return {"error": "invalid-face-source-binding"}
                    source_id = region.get("faceSourceId")
                    if not source_id or str(source_id) not in source_map:
                        response.status = 400
                        return {"error": "invalid-face-source-binding"}
                res, err = AsyncTask.run(
                    lambda: swap_face_regions_by_sources(input_path, source_map, regions),
                    task_id=task_id,
                )
            else:
                fallback_face = next(iter(source_map.values()))
                res, err = AsyncTask.run(
                    lambda: swap_face(input_path, fallback_face),
                    task_id=task_id,
                )
        else:
            if not target_face_id:
                response.status = 400
                return {"error": "missing-params"}
            target_face_path = _get_library_path(str(target_face_id))
            if not target_face_path:
                response.status = 400
                return {"error": "file-not-found"}
            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code="unsupported-image-format",
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}
            if regions:
                res, err = AsyncTask.run(
                    lambda: swap_face_regions(input_path, target_face_path, regions),
                    task_id=task_id,
                )
            else:
                res, err = AsyncTask.run(
                    lambda: swap_face(input_path, target_face_path),
                    task_id=task_id,
                )
        if res:
            result_id = _register_result(res, [input_path, res])
            return {"resultFileId": result_id, "resultUrl": f"/api/file/{result_id}"}
        response.status = 500
        return {"error": _simplify_task_error(err)}
    except Exception as e:
        print("[ERROR] create_task failed:", str(e), "\n", traceback.format_exc())
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.route("/api/task/video", method=["POST", "OPTIONS"])
def create_video_task():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}
    task_id = None
    try:
        body = request.json or {}
        task_id = body.get("id")
        input_id = body.get("inputFileId")
        target_face_id = body.get("targetFaceId")
        regions = body.get("regions")
        face_sources = body.get("faceSources")
        key_frame_ms = body.get("keyFrameMs", 0)
        use_gpu = body.get("useGpu", False)
        has_face_sources = "faceSources" in body

        if not all([task_id, input_id]):
            response.status = 400
            return {"error": "missing-params"}
        input_path = _get_upload_path(input_id)
        if not input_path:
            response.status = 400
            return {"error": "file-not-found"}
        try:
            _validate_file(
                input_path,
                ALLOWED_VIDEO_EXTS,
                missing_code="unsupported-video-format",
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {"error": _simplify_task_error(e)}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status = 400
                return {"error": "missing-face-sources"}
            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status = 400
                    return {"error": "missing-face-sources"}
                source_id = source.get("id")
                source_path = _get_library_path(str(source_id))
                if not source_id or not source_path:
                    response.status = 400
                    return {"error": "missing-face-sources"}
                source_map[str(source_id)] = source_path
            if not isinstance(regions, list) or len(regions) == 0:
                response.status = 400
                return {"error": "invalid-face-source-binding"}
            for region in regions:
                if not isinstance(region, dict):
                    response.status = 400
                    return {"error": "invalid-face-source-binding"}
                source_id = region.get("faceSourceId")
                if not source_id or str(source_id) not in source_map:
                    response.status = 400
                    return {"error": "invalid-face-source-binding"}
            try:
                key_frame_ms = int(float(key_frame_ms or 0))
            except (TypeError, ValueError):
                key_frame_ms = 0
            key_frame_ms = max(0, key_frame_ms)

        _set_video_task_progress(
            task_id,
            status="running",
            progress=0,
            etaSeconds=None,
            frameCount=0,
            totalFrames=0,
            error=None,
            resultFileId=None,
            resultUrl=None,
        )

        def _on_stage(stage: str):
            print(f"[INFO] 视频处理阶段: {stage}")

        def _on_progress(frame_count: int, total_frames: int, elapsed_seconds: float):
            progress = 0.0
            eta_seconds = None
            if total_frames and total_frames > 0:
                progress = max(0.0, min(100.0, frame_count / total_frames * 100.0))
                if frame_count > 0 and elapsed_seconds > 0:
                    frames_remaining = total_frames - frame_count
                    processing_speed = frame_count / elapsed_seconds
                    eta_seconds = max(0, int(frames_remaining / processing_speed))
            _set_video_task_progress(
                task_id,
                status="running",
                progress=round(progress, 2),
                etaSeconds=eta_seconds,
                frameCount=frame_count,
                totalFrames=total_frames,
                error=None,
            )

        if has_face_sources:
            task_callable = lambda: swap_face_video_by_sources(
                input_path,
                source_map,
                regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
            )
        else:
            if not target_face_id:
                response.status = 400
                return {"error": "missing-params"}
            target_face_path = _get_library_path(str(target_face_id))
            if not target_face_path:
                response.status = 400
                return {"error": "file-not-found"}
            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code="unsupported-image-format",
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}
            task_callable = lambda: swap_face_video(
                input_path,
                target_face_path,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
            )

        def _on_completion(res, err):
            if res:
                result_id = _register_result(res, [input_path, res])
                _set_video_task_progress(
                    task_id,
                    status="success",
                    progress=100,
                    etaSeconds=0,
                    error=None,
                    resultFileId=result_id,
                    resultUrl=f"/api/file/{result_id}",
                )
            else:
                final_error = _simplify_task_error(err)
                _set_video_task_progress(
                    task_id,
                    status="failed",
                    error=final_error,
                    etaSeconds=None,
                )

        try:
            AsyncTask.run_async(
                task_callable, task_id=task_id, on_completion=_on_completion
            )
        except Exception as e:
            _set_video_task_progress(
                task_id,
                status="failed",
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
            response.status = 500
            return {"error": _simplify_task_error(e)}
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        print("[ERROR] create_video_task failed:", str(e), "\n", traceback.format_exc())
        if task_id:
            _set_video_task_progress(
                task_id,
                status="failed",
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.get("/api/task/video/progress/<task_id>")
def get_video_task_progress(task_id):
    return _get_video_task_progress(task_id)


@app.delete("/api/task/<task_id>")
def cancel_task(task_id):
    AsyncTask.cancel(task_id)
    _set_video_task_progress(
        task_id,
        status="cancelled",
        etaSeconds=None,
        error="cancelled",
    )
    return {"success": True}


@app.get("/")
def web_index():
    if os.path.isdir(DIST_DIR):
        return static_file("index.html", root=DIST_DIR)
    response.status = 404
    return "web dist not found"


@app.get("/<filepath:path>")
def web_assets(filepath):
    if os.path.isdir(DIST_DIR):
        candidate = os.path.join(DIST_DIR, filepath)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return static_file(filepath, root=DIST_DIR)
        return static_file("index.html", root=DIST_DIR)
    response.status = 404
    return "web dist not found"


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8033"))
    print(f"[WEB] starting server on {host}:{port}")
    httpd = make_server(
        host,
        port,
        app,
        server_class=_ThreadingWSGIServer,
        handler_class=WSGIRequestHandler,
    )
    httpd.serve_forever()