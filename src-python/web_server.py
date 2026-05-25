import json
import mimetypes
import os
import re
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
    get_gpu_acceleration_modes,
    load_models,
    swap_face,
    swap_face_deep,
    swap_face_regions,
    swap_face_regions_by_sources,
    swap_face_video,
    swap_face_video_by_sources,
    swap_face_video_deep,
)
from magic.task_config import (
    build_video_task_config_token,
    compute_file_sha256,
    get_expected_face_source_sha256_map,
    get_expected_input_video_sha256,
    get_expected_target_face_sha256,
    get_expected_target_faces_sha256_map,
    parse_video_task_config_token,
    verify_file_sha256,
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
WEB_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "web"))
UPLOADS_DIR = os.path.abspath(os.path.join(WEB_DATA_DIR, "uploads"))
LIBRARY_DIR = os.path.abspath(os.path.join(WEB_DATA_DIR, "library"))
CONFIG_PATH = os.path.join(WEB_DATA_DIR, "config.json")
DIST_DIR = os.path.join(BASE_DIR, "dist-web")

TOKEN_TTL_SECONDS = 7 * 24 * 3600

# Upload/cache TTL and size limits
UPLOAD_TTL_SECONDS = 24 * 3600
RESULT_TTL_SECONDS = 4 * 3600
PROGRESS_TTL_SECONDS = 6 * 3600
MAX_UPLOAD_BYTES_IMAGE = 50 * 1024 * 1024
MAX_UPLOAD_BYTES_VIDEO = 2 * 1024 * 1024 * 1024
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._\-]+")

TOKENS: Dict[str, float] = {}
TOKENS_LOCK = threading.RLock()

UPLOADS: Dict[str, Dict[str, object]] = {}
UPLOADS_LOCK = threading.RLock()

RESULTS: Dict[str, Dict[str, object]] = {}
RESULTS_LOCK = threading.RLock()

VIDEO_TASK_PROGRESS: Dict[str, Dict[str, object]] = {}
VIDEO_TASK_PROGRESS_LOCK = threading.RLock()
VIDEO_TASK_CANCELLED = set()
VIDEO_TASK_CANCELLED_LOCK = threading.RLock()

VIDEO_TASK_CONFIGS: Dict[str, Dict[str, object]] = {}
VIDEO_TASK_CONFIGS_LOCK = threading.RLock()
VIDEO_TASK_CONFIG_TTL_SECONDS = 7 * 24 * 3600
VIDEO_TASK_CONFIG_TOKEN_PREFIX = "cfg1"
VIDEO_TASK_CONFIG_SECRET = os.environ.get(
    "VIDEO_TASK_CONFIG_SECRET", "magic-mirror-config-secret"
)

_LIBRARY_CACHE_LOCK = threading.RLock()
_LIBRARY_CACHE_MTIME: Optional[int] = None
_LIBRARY_CACHE_ITEMS: List[Dict[str, str]] = []

def _ensure_dirs():
    """Create required data directories if they do not exist."""
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(LIBRARY_DIR, exist_ok=True)


_ensure_dirs()


def _load_config() -> dict:
    """Load or create the server configuration file with default password."""
    if not os.path.exists(CONFIG_PATH):
        _save_config({"password": "123456"})
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg: dict) -> None:
    """Persist the server configuration to disk."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


    """Generate a new authentication token."""
def _issue_token() -> str:
    token = uuid.uuid4().hex
    with TOKENS_LOCK:
        TOKENS[token] = time.time()
    return token


    """Remove expired authentication tokens."""
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


    """Extract the authentication token from the request."""
def _extract_token() -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Token") or request.query.get("token")


    """Validate the request token and reject unauthorized calls."""
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
        TOKENS[token] = time.time()
    return True


    """Get the file extension from a path."""
def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


    """Validate a file exists and has an allowed extension."""
def _validate_file(path: str, allowed_exts: set, *, missing_code: str):
    if not path:
        raise RuntimeError("missing-params")
    if not os.path.exists(path):
        raise FileNotFoundError("file-not-found")
    if _ext(path) not in allowed_exts:
        raise RuntimeError(missing_code)


    """Simplify a task error to a human-readable string."""
def _simplify_task_error(err: object) -> str:
    msg = (str(err) if err is not None else "").lower()
    codes = [
        "missing-params",
        "missing-face-sources",
        "invalid-face-source-binding",
        "face-source-not-found",
        "file-not-found",
        "file-too-large",
        "invalid-path",
        "unsupported-image-format",
        "unsupported-video-format",
        "unsupported-file-format",
        "image-decode-failed",
        "no-face-detected",
        "no-face-in-selected-regions",
        "swap-failed",
        "video-open-failed",
        "video-write-failed",
        "video-output-missing",
        "audio-mux-failed",
        "video-frame-read-failed",
        "output-write-failed",
        "invalid-regions",
        "config-mismatch",
        "config-not-found",
    ]
    for code in codes:
        if code in msg:
            return code
    return "internal"


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename by removing unsafe characters and limiting length."""
    base = os.path.basename(name or "upload")
    base = base.replace(" ", "_")
    cleaned = SAFE_FILENAME_PATTERN.sub("_", base).strip("._-") or "upload"
    if len(cleaned) > 200:
        root, ext = os.path.splitext(cleaned)
        cleaned = root[: 200 - len(ext)] + ext
    return cleaned


def _is_path_within(parent: str, child: str) -> bool:
    """Check child path is within parent directory (path traversal defense)."""
    try:
        parent_abs = os.path.abspath(parent)
        child_abs = os.path.abspath(child)
        return os.path.commonpath([parent_abs, child_abs]) == parent_abs
    except (ValueError, OSError):
        return False


def _check_upload_size(upload_file, max_bytes: int) -> bool:
    """Check upload file size without reading entire content."""
    raw = getattr(upload_file, "file", None)
    if raw is None:
        return True
    try:
        current = raw.tell()
        raw.seek(0, 2)
        size = raw.tell()
        raw.seek(current)
        return size <= max_bytes
    except (OSError, AttributeError):
        return True


    """Save an uploaded file to the destination directory."""
def _save_upload(upload_file, dest_dir: str, *, max_bytes: Optional[int] = None):
    if max_bytes is not None and not _check_upload_size(upload_file, max_bytes):
        raise RuntimeError("file-too-large")
    filename = _sanitize_filename(upload_file.filename)
    ext = os.path.splitext(filename)[1].lower()
    file_id = uuid.uuid4().hex
    safe_name = f"{file_id}{ext}" if ext else file_id
    save_path = os.path.abspath(os.path.join(dest_dir, safe_name))
    if not _is_path_within(dest_dir, save_path):
        raise RuntimeError("invalid-path")
    upload_file.save(save_path, overwrite=True)
    return file_id, save_path, safe_name


    """Register an uploaded file for later retrieval."""
def _register_upload(file_id: str, path: str, kind: str) -> None:
    with UPLOADS_LOCK:
        UPLOADS[file_id] = {"path": path, "kind": kind, "createdAt": time.time()}


    """Deep clone a JSON-serializable payload."""
def _clone_json_payload(payload):
    return json.loads(json.dumps(payload, ensure_ascii=False))


    """Build a signed config token for a video task."""
def _build_video_task_config_token(payload: Dict[str, object]) -> str:
    return build_video_task_config_token(payload, VIDEO_TASK_CONFIG_SECRET)


    """Parse and verify a video task config token."""
def _parse_video_task_config_token(config_id: str) -> Optional[Dict[str, object]]:
    return parse_video_task_config_token(
        str(config_id),
        VIDEO_TASK_CONFIG_SECRET,
        legacy_ttl_seconds=VIDEO_TASK_CONFIG_TTL_SECONDS,
    )


    """Register a result file for download and cleanup."""
def _register_result(result_path: str, delete_paths: List[str]) -> str:
    result_id = uuid.uuid4().hex
    with RESULTS_LOCK:
        RESULTS[result_id] = {
            "path": result_path,
            "delete_paths": delete_paths,
            "name": os.path.basename(result_path),
            "createdAt": time.time(),
        }
    return result_id


    """Get the file path for an uploaded file."""
def _get_upload_path(file_id: str) -> Optional[str]:
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        return item.get("path") if item else None


    """Get the file kind for an uploaded file."""
def _get_upload_kind(file_id: str) -> Optional[str]:
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        return item.get("kind") if item else None


    """Get info about a registered result."""
def _get_result_info(file_id: str) -> Optional[Dict[str, object]]:
    with RESULTS_LOCK:
        info = RESULTS.get(file_id)
        return info.copy() if info else None


    """Remove an upload registration by path."""
def _remove_upload_by_path(path: str) -> None:
    with UPLOADS_LOCK:
        to_remove = [key for key, item in UPLOADS.items() if item.get("path") == path]
        for key in to_remove:
            UPLOADS.pop(key, None)


    """Safely delete a file, ignoring errors."""
def _safe_delete(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ─── Garbage Collection ───────────────────────────────────────────────────────


def _cleanup_expired_uploads() -> None:
    """Remove upload records and files older than UPLOAD_TTL_SECONDS."""
    now = time.time()
    expired_paths: List[str] = []
    with UPLOADS_LOCK:
        expired_ids = [
            fid
            for fid, entry in UPLOADS.items()
            if now - float(entry.get("createdAt", now)) > UPLOAD_TTL_SECONDS
        ]
        for fid in expired_ids:
            entry = UPLOADS.pop(fid, None)
            if entry:
                p = entry.get("path")
                if isinstance(p, str):
                    expired_paths.append(p)
    for p in expired_paths:
        _safe_delete(p)


def _cleanup_expired_results() -> None:
    """Remove result records and files older than RESULT_TTL_SECONDS."""
    now = time.time()
    expired_delete_lists: List[List[str]] = []
    with RESULTS_LOCK:
        expired_ids = [
            rid
            for rid, entry in RESULTS.items()
            if now - float(entry.get("createdAt", now)) > RESULT_TTL_SECONDS
        ]
        for rid in expired_ids:
            entry = RESULTS.pop(rid, None)
            if entry:
                expired_delete_lists.append(list(entry.get("delete_paths") or []))
    for paths in expired_delete_lists:
        for p in paths:
            _safe_delete(p)


def _cleanup_expired_progress() -> None:
    """Remove finished task progress records older than PROGRESS_TTL_SECONDS."""
    now = time.time()
    finished_states = {"success", "failed", "cancelled"}
    with VIDEO_TASK_PROGRESS_LOCK:
        to_remove = []
        for task_id in list(VIDEO_TASK_PROGRESS.keys()):
            state = VIDEO_TASK_PROGRESS.get(task_id) or {}
            status = state.get("status")
            finished_at = state.get("_finishedAt")
            if status in finished_states:
                if finished_at is None:
                    state["_finishedAt"] = now
                    VIDEO_TASK_PROGRESS[task_id] = state
                elif now - float(finished_at) > PROGRESS_TTL_SECONDS:
                    to_remove.append(task_id)
        for task_id in to_remove:
            VIDEO_TASK_PROGRESS.pop(task_id, None)
    if to_remove:
        with VIDEO_TASK_CANCELLED_LOCK:
            for task_id in to_remove:
                VIDEO_TASK_CANCELLED.discard(task_id)


def _cleanup_orphan_upload_files() -> None:
    """Scan uploads dir and remove orphan files not tracked in UPLOADS."""
    if not os.path.isdir(UPLOADS_DIR):
        return
    now = time.time()
    with UPLOADS_LOCK:
        known_paths = {
            os.path.abspath(str(entry.get("path", "")))
            for entry in UPLOADS.values()
            if entry.get("path")
        }
    try:
        for name in os.listdir(UPLOADS_DIR):
            full = os.path.abspath(os.path.join(UPLOADS_DIR, name))
            if not os.path.isfile(full):
                continue
            if full in known_paths:
                continue
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if now - mtime > UPLOAD_TTL_SECONDS:
                _safe_delete(full)
    except OSError:
        pass


_LAST_GC_AT = 0.0
_GC_LOCK = threading.RLock()
_GC_INTERVAL_SECONDS = 5 * 60


def _maybe_run_gc() -> None:
    """Throttled GC entry point, triggered by requests."""
    global _LAST_GC_AT
    now = time.time()
    with _GC_LOCK:
        if now - _LAST_GC_AT < _GC_INTERVAL_SECONDS:
            return
        _LAST_GC_AT = now
    try:
        _cleanup_expired_uploads()
        _cleanup_expired_results()
        _cleanup_expired_progress()
        _cleanup_orphan_upload_files()
    except Exception:
        print("[WEB] GC failed:", traceback.format_exc())


# ─── End GC ───────────────────────────────────────────────────────────────────


    """Invalidate the face library cache."""
def _invalidate_library_cache() -> None:
    global _LIBRARY_CACHE_MTIME, _LIBRARY_CACHE_ITEMS
    with _LIBRARY_CACHE_LOCK:
        _LIBRARY_CACHE_MTIME = None
        _LIBRARY_CACHE_ITEMS = []


    """List all items in the face library."""
def _list_library_items() -> List[Dict[str, str]]:
    if not os.path.isdir(LIBRARY_DIR):
        _invalidate_library_cache()
        return []

    try:
        dir_mtime = os.stat(LIBRARY_DIR).st_mtime_ns
    except OSError:
        _invalidate_library_cache()
        return []

    with _LIBRARY_CACHE_LOCK:
        if _LIBRARY_CACHE_MTIME == dir_mtime:
            return [item.copy() for item in _LIBRARY_CACHE_ITEMS]

    items: List[Dict[str, str]] = []
    try:
        entries = sorted(os.scandir(LIBRARY_DIR), key=lambda entry: entry.name)
        for entry in entries:
            if not entry.is_file():
                continue
            if _ext(entry.name) not in ALLOWED_IMAGE_EXTS:
                continue
            items.append(
                {
                    "id": entry.name,
                    "name": entry.name,
                    "url": f"/api/library/{entry.name}",
                }
            )
    except OSError:
        return []

    with _LIBRARY_CACHE_LOCK:
        _LIBRARY_CACHE_MTIME = dir_mtime
        _LIBRARY_CACHE_ITEMS = [item.copy() for item in items]

    return items


    """Get the file path for a library item."""
def _get_library_path(item_id: str) -> Optional[str]:
    if not item_id:
        return None
    path = os.path.join(LIBRARY_DIR, os.path.basename(item_id))
    if not os.path.exists(path):
        return None
    if _ext(path) not in ALLOWED_IMAGE_EXTS:
        return None
    return path


    """Update the progress of a video task."""
def _set_video_task_progress(task_id: str, **updates):
    status = updates.get("status")
    if status in {"success", "failed", "cancelled"} and "_finishedAt" not in updates:
        updates["_finishedAt"] = time.time()
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id, {})
        state.update(updates)
        VIDEO_TASK_PROGRESS[task_id] = state


    """Get the current progress of a video task."""
def _get_video_task_progress(task_id: str):
    _maybe_run_gc()
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id)
        if not state:
            return {
                "status": "idle",
                "progress": 0,
                "etaSeconds": None,
                "stage": None,
            }
        public_state = state.copy()
    public_state.pop("_finishedAt", None)
    return public_state


    """Mark a video task as cancelled."""
def _mark_video_task_cancelled(task_id: str):
    with VIDEO_TASK_CANCELLED_LOCK:
        VIDEO_TASK_CANCELLED.add(task_id)


    """Clear the cancelled status of a video task."""
def _clear_video_task_cancelled(task_id: str):
    with VIDEO_TASK_CANCELLED_LOCK:
        VIDEO_TASK_CANCELLED.discard(task_id)


    """Check if a video task has been cancelled."""
def _is_video_task_cancelled(task_id: str) -> bool:
    with VIDEO_TASK_CANCELLED_LOCK:
        return task_id in VIDEO_TASK_CANCELLED


    """Run a video task asynchronously in a background thread."""
def _run_video_task_async(task_id: str, task_callable, on_completion):
    def _worker():
        res = None
        err = None
        try:
            res = task_callable()
        except Exception as e:
            err = e
        try:
            on_completion(res, err)
        except Exception:
            print("[ERROR] web video task completion callback failed:\n", traceback.format_exc())
        finally:
            _clear_video_task_cancelled(task_id)

    thread = threading.Thread(target=_worker, name=f"WebVideoTask-{task_id}", daemon=True)
    thread.start()


    """Clean up expired video task configurations."""
def _cleanup_video_task_configs() -> None:
    now = time.time()
    with VIDEO_TASK_CONFIGS_LOCK:
        expired = [
            config_id
            for config_id, item in VIDEO_TASK_CONFIGS.items()
            if now - float(item.get("createdAt", 0)) > VIDEO_TASK_CONFIG_TTL_SECONDS
        ]
        for config_id in expired:
            VIDEO_TASK_CONFIGS.pop(config_id, None)


    """Store a video task configuration."""
def _store_video_task_config(payload: Dict[str, object], config_id: Optional[str] = None) -> str:
    _cleanup_video_task_configs()
    next_id = str(config_id or _build_video_task_config_token(payload))
    with VIDEO_TASK_CONFIGS_LOCK:
        VIDEO_TASK_CONFIGS[next_id] = {
            "createdAt": time.time(),
            "config": _clone_json_payload(payload),
        }
    return next_id


    """Get a stored video task configuration."""
def _get_video_task_config(config_id: str) -> Optional[Dict[str, object]]:
    if not config_id:
        return None
    _cleanup_video_task_configs()
    with VIDEO_TASK_CONFIGS_LOCK:
        item = VIDEO_TASK_CONFIGS.get(str(config_id))
        if item:
            item["createdAt"] = time.time()
            config = item.get("config")
            if isinstance(config, dict):
                return _clone_json_payload(config)

    return _parse_video_task_config_token(str(config_id))


    """Extract the file path from a stored entry."""
def _extract_stored_path(file_entry):
    if isinstance(file_entry, str) and file_entry:
        return file_entry
    if isinstance(file_entry, dict):
        path = file_entry.get("path")
        if isinstance(path, str) and path:
            return path
    return None


    """Extract face source paths from stored entries."""
def _extract_stored_face_sources(face_sources):
    if not isinstance(face_sources, list):
        return None

    resolved = []
    for source in face_sources:
        if not isinstance(source, dict):
            return None
        source_id = source.get("id")
        source_path = _extract_stored_path(source)
        if source_id is None or not source_path:
            return None
        resolved.append({"id": str(source_id), "path": source_path})

    return resolved or None


    """Resolve a face reference to a file path."""
def _resolve_face_reference_path(face_ref: str) -> Optional[str]:
    if not face_ref:
        return None
    library_path = _get_library_path(str(face_ref))
    if library_path:
        return library_path
    if os.path.exists(face_ref):
        return face_ref
    return None


    """Resolve target face items to file paths."""
def _resolve_target_face_items(target_faces) -> List[Dict[str, str]]:
    if not isinstance(target_faces, list) or len(target_faces) == 0:
        raise RuntimeError("missing-params")

    resolved: List[Dict[str, str]] = []
    for index, target in enumerate(target_faces):
        target_id = f"target-{index + 1}"
        target_ref = None

        if isinstance(target, str):
            target_ref = target
        elif isinstance(target, dict):
            target_id = str(target.get("id") or target_id)
            target_ref = target.get("path") or target.get("id")
        else:
            raise RuntimeError("missing-params")

        if not target_ref:
            raise RuntimeError("missing-params")

        target_path = _resolve_face_reference_path(str(target_ref))
        if not target_path:
            raise FileNotFoundError("file-not-found")

        _validate_file(
            target_path,
            ALLOWED_IMAGE_EXTS,
            missing_code="unsupported-image-format",
        )
        resolved.append({"id": str(target_id), "path": target_path})

    return resolved


    """Build the config payload for a video task."""
def _build_video_task_config_payload(
    *,
    input_path: str,
    target_face_path: Optional[str] = None,
    target_face_id: Optional[str] = None,
    target_faces: Optional[List[Dict[str, object]]] = None,
    deep_swap_mode: bool = False,
    segment_duration_sec: int = 12,
    segment_overlap_frames: int = 6,
    face_sources: Optional[List[Dict[str, object]]] = None,
    source_map: Optional[Dict[str, str]] = None,
    regions=None,
    key_frame_ms: int = 0,
    use_gpu: bool = False,
    gpu_provider: str = "auto",
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "inputVideo": {
            "path": input_path,
            "sha256": compute_file_sha256(input_path),
        },
        "regions": regions if isinstance(regions, list) else None,
        "keyFrameMs": max(0, int(key_frame_ms or 0)),
        "useGpu": bool(use_gpu),
        "gpuProvider": str(gpu_provider or "auto").strip().lower(),
    }

    if isinstance(face_sources, list) and len(face_sources) > 0:
        normalized_sources = []
        source_map = source_map or {}
        for source in face_sources:
            if not isinstance(source, dict):
                raise RuntimeError("missing-face-sources")
            source_id = source.get("id")
            if source_id is None:
                raise RuntimeError("missing-face-sources")
            source_path = source_map.get(str(source_id))
            if not source_path:
                raise RuntimeError("missing-face-sources")
            normalized_sources.append(
                {
                    "id": str(source_id),
                    "path": source_path,
                    "sha256": compute_file_sha256(source_path),
                }
            )
        payload["faceSources"] = normalized_sources
    elif isinstance(target_faces, list) and len(target_faces) > 0:
        normalized_targets = []
        for idx, target in enumerate(target_faces):
            if not isinstance(target, dict):
                raise RuntimeError("missing-params")
            target_id = target.get("id")
            target_path = target.get("path")
            if not target_path:
                raise RuntimeError("missing-params")
            normalized_targets.append(
                {
                    "id": str(target_id or f"target-{idx + 1}"),
                    "path": target_path,
                    "sha256": compute_file_sha256(target_path),
                }
            )
        payload["targetFaces"] = normalized_targets
        payload["deepSwapMode"] = bool(deep_swap_mode)
        payload["segmentDurationSec"] = max(1, int(segment_duration_sec or 1))
        payload["segmentOverlapFrames"] = max(0, int(segment_overlap_frames or 0))
    else:
        if not target_face_path:
            raise RuntimeError("missing-params")
        if target_face_id is not None:
            payload["targetFaceId"] = str(target_face_id)
        payload["targetFace"] = {
            "path": target_face_path,
            "sha256": compute_file_sha256(target_face_path),
        }

    return payload


    """Ensure a video task config matches expectations."""
def _ensure_video_task_config_matches(
    config: Dict[str, object],
    input_path: str,
    *,
    target_face_path: Optional[str] = None,
    target_face_map: Optional[Dict[str, str]] = None,
    source_map: Optional[Dict[str, str]] = None,
) -> None:
    expected_input_sha256 = get_expected_input_video_sha256(config)
    if expected_input_sha256 and not verify_file_sha256(input_path, expected_input_sha256):
        raise RuntimeError("config-mismatch")

    expected_target_sha256 = get_expected_target_face_sha256(config)
    if expected_target_sha256:
        if not target_face_path or not verify_file_sha256(target_face_path, expected_target_sha256):
            raise RuntimeError("config-mismatch")

    expected_target_face_sha256_map = get_expected_target_faces_sha256_map(config)
    if expected_target_face_sha256_map:
        if not isinstance(target_face_map, dict):
            raise RuntimeError("config-mismatch")
        for target_id, expected_sha256 in expected_target_face_sha256_map.items():
            target_path = target_face_map.get(str(target_id))
            if not target_path or not verify_file_sha256(target_path, expected_sha256):
                raise RuntimeError("config-mismatch")

    expected_source_sha256_map = get_expected_face_source_sha256_map(config)
    if expected_source_sha256_map:
        if not isinstance(source_map, dict):
            raise RuntimeError("config-mismatch")
        for source_id, expected_sha256 in expected_source_sha256_map.items():
            source_path = source_map.get(str(source_id))
            if not source_path or not verify_file_sha256(source_path, expected_sha256):
                raise RuntimeError("config-mismatch")


    """Clean up a result and its associated files."""
def _cleanup_result(result_id: str, delete_paths: List[str]) -> None:
    with RESULTS_LOCK:
        RESULTS.pop(result_id, None)
    for path in delete_paths:
        _safe_delete(path)
        _remove_upload_by_path(path)


    """Stream a file and clean up after download."""
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
    """Configure CORS headers for all responses."""
def enable_cors():
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header(
        "Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS"
    )
    response.set_header(
        "Access-Control-Allow-Headers", "Authorization,Content-Type,X-Token"
    )


@app.hook("before_request")
def _trigger_gc_hook():
    """Throttled memory/disk cleanup, triggered by incoming requests."""
    _maybe_run_gc()


@app.route("/api/<path:path>", method=["OPTIONS"])
    """Handle CORS preflight OPTIONS requests."""
def handle_options(path):
    response.status = 200
    return {}


@app.get("/api/status")
    """Return the server status and available features."""
def status():
    return {"status": "running"}


@app.post("/api/prepare")
    """Prepare a file for face detection."""
def prepare():
    if request.method == "OPTIONS":
        return {}
    return {"success": load_models()}


@app.post("/api/login")
    """Authenticate with password and return a token."""
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
    """Update the server password."""
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
    """List all items in the face library."""
def list_library():
    if not _require_auth():
        return {"error": "unauthorized"}
    return {"items": _list_library_items()}


@app.post("/api/library/upload")
    """Upload a new face to the library."""
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
    try:
        _, save_path, safe_name = _save_upload(
            upload_file, LIBRARY_DIR, max_bytes=MAX_UPLOAD_BYTES_IMAGE
        )
    except RuntimeError as e:
        response.status = 400
        return {"error": _simplify_task_error(e)}
    _invalidate_library_cache()
    return {
        "id": safe_name,
        "name": safe_name,
        "url": f"/api/library/{safe_name}",
    }


@app.get("/api/library/<filename>")
    """Retrieve a face library file by filename."""
def get_library_file(filename):
    safe_name = os.path.basename(filename or "")
    path = os.path.abspath(os.path.join(LIBRARY_DIR, safe_name))
    if not _is_path_within(LIBRARY_DIR, path) or not os.path.exists(path):
        response.status = 404
        return {"error": "file-not-found"}
    if _ext(path) not in ALLOWED_IMAGE_EXTS:
        response.status = 404
        return {"error": "file-not-found"}
    return static_file(os.path.basename(path), root=os.path.dirname(path))


@app.post("/api/upload")
    """Upload a file for processing."""
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
        max_bytes = MAX_UPLOAD_BYTES_IMAGE
    elif ext in ALLOWED_VIDEO_EXTS:
        kind = "video"
        max_bytes = MAX_UPLOAD_BYTES_VIDEO
    else:
        response.status = 400
        return {"error": "unsupported-file-format"}
    try:
        file_id, save_path, _ = _save_upload(
            upload_file, UPLOADS_DIR, max_bytes=max_bytes
        )
    except RuntimeError as e:
        response.status = 400
        return {"error": _simplify_task_error(e)}
    _register_upload(file_id, save_path, kind)
    return {
        "fileId": file_id,
        "url": f"/api/file/{file_id}",
        "type": kind,
        "name": upload_file.filename or "upload",
    }


@app.get("/api/file/<file_id>")
    """Retrieve an uploaded file by ID."""
def get_file(file_id):
    if not _require_auth():
        return {"error": "unauthorized"}
    path = _get_upload_path(file_id)
    if not path:
        info = _get_result_info(file_id)
        if info:
            path = info.get("path")
    if not path or not os.path.exists(path):
        response.status = 404
        return {"error": "file-not-found"}
    abs_path = os.path.abspath(str(path))
    if not (
        _is_path_within(UPLOADS_DIR, abs_path)
        or _is_path_within(LIBRARY_DIR, abs_path)
        or _is_path_within(WEB_DATA_DIR, abs_path)
    ):
        response.status = 404
        return {"error": "file-not-found"}
    response.set_header("Cache-Control", "no-store")
    return static_file(os.path.basename(abs_path), root=os.path.dirname(abs_path))


@app.get("/api/download/<file_id>")
    """Download a result file by ID."""
def download_file(file_id):
    if not _require_auth():
        return {"error": "unauthorized"}
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
    """Detect faces in an uploaded image."""
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
    """Detect faces in a video file."""
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
    """Create a new face swap task."""
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
        target_faces = body.get("targetFaces")
        deep_swap_mode = bool(body.get("deepSwapMode", False))
        has_face_sources = "faceSources" in body
        has_target_faces = "targetFaces" in body or deep_swap_mode

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

        if has_face_sources and has_target_faces:
            response.status = 400
            return {"error": "missing-params"}

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
        elif has_target_faces:
            try:
                target_face_items = _resolve_target_face_items(target_faces)
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

            res, err = AsyncTask.run(
                lambda: swap_face_deep(
                    input_path,
                    [item["path"] for item in target_face_items],
                    regions=regions,
                ),
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


@app.route("/api/task/video/gpu-modes", method=["GET", "OPTIONS"])
    """Return available GPU acceleration modes."""
def get_video_gpu_modes():
    if request.method == "OPTIONS":
        return {}
    if not _require_auth():
        return {"error": "unauthorized"}

    try:
        return get_gpu_acceleration_modes()
    except Exception as e:
        response.status = 500
        return {
            "modes": [{"id": "cpu", "name": "CPU"}],
            "availableProviders": [],
            "error": _simplify_task_error(e),
        }


@app.route("/api/task/video", method=["POST", "OPTIONS"])
    """Create a new video face swap task."""
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
        target_faces = body.get("targetFaces")
        regions = body.get("regions")
        face_sources = body.get("faceSources")
        deep_swap_mode = bool(body.get("deepSwapMode", False))
        segment_duration_sec = body.get("segmentDurationSec", 12)
        segment_overlap_frames = body.get("segmentOverlapFrames", 6)
        key_frame_ms = body.get("keyFrameMs", 0)
        use_gpu = body.get("useGpu", False)
        gpu_provider = str(body.get("gpuProvider", "auto") or "auto").strip().lower()
        config_id = body.get("configId")
        generate_config_id = bool(body.get("generateConfigId", False))
        dry_run_config_only = bool(body.get("dryRunConfigOnly", False))

        stored_config = None
        stored_target_face_path = None
        if config_id:
            stored_config = _get_video_task_config(str(config_id))
            if not isinstance(stored_config, dict):
                response.status = 400
                return {"error": "config-not-found"}

            stored_target_face_path = _extract_stored_path(stored_config.get("targetFace"))

            if target_face_id is None:
                target_face_id = stored_config.get("targetFaceId")
            if target_faces is None and "targetFaces" in stored_config:
                target_faces = _extract_stored_face_sources(stored_config.get("targetFaces"))
            if face_sources is None and "faceSources" in stored_config:
                face_sources = stored_config.get("faceSources")
            if regions is None and "regions" in stored_config:
                regions = stored_config.get("regions")
            if "deepSwapMode" not in body:
                deep_swap_mode = bool(stored_config.get("deepSwapMode", deep_swap_mode))
            if "segmentDurationSec" not in body:
                segment_duration_sec = stored_config.get(
                    "segmentDurationSec", segment_duration_sec
                )
            if "segmentOverlapFrames" not in body:
                segment_overlap_frames = stored_config.get(
                    "segmentOverlapFrames", segment_overlap_frames
                )
            if "keyFrameMs" not in body:
                key_frame_ms = stored_config.get("keyFrameMs", key_frame_ms)
            if "useGpu" not in body:
                use_gpu = bool(stored_config.get("useGpu", use_gpu))
            if "gpuProvider" not in body:
                gpu_provider = str(
                    stored_config.get("gpuProvider", gpu_provider) or gpu_provider
                ).strip().lower()

        if gpu_provider in ("dml", "directml"):
            gpu_provider = "directml"
        elif gpu_provider == "cuda":
            gpu_provider = "cuda"
        elif gpu_provider == "cpu":
            gpu_provider = "cpu"
        else:
            gpu_provider = "auto"

        if gpu_provider == "cpu":
            use_gpu = False
        elif gpu_provider in ("directml", "cuda"):
            use_gpu = True

        try:
            key_frame_ms = int(float(key_frame_ms or 0))
        except (TypeError, ValueError):
            key_frame_ms = 0
        key_frame_ms = max(0, key_frame_ms)

        try:
            segment_duration_sec = int(float(segment_duration_sec or 12))
        except (TypeError, ValueError):
            segment_duration_sec = 12
        segment_duration_sec = max(1, segment_duration_sec)

        try:
            segment_overlap_frames = int(float(segment_overlap_frames or 6))
        except (TypeError, ValueError):
            segment_overlap_frames = 6
        segment_overlap_frames = max(0, segment_overlap_frames)

        has_face_sources = face_sources is not None
        has_target_faces = (target_faces is not None) or deep_swap_mode

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

        source_map: Optional[Dict[str, str]] = None
        target_face_path: Optional[str] = None
        target_face_map: Optional[Dict[str, str]] = None
        target_face_items: Optional[List[Dict[str, str]]] = None

        if has_face_sources and has_target_faces:
            response.status = 400
            return {"error": "missing-params"}

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
        elif has_target_faces:
            try:
                target_face_items = _resolve_target_face_items(target_faces)
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

            target_face_map = {
                str(item.get("id")): str(item.get("path"))
                for item in (target_face_items or [])
            }
        else:
            if target_face_id is not None:
                target_face_path = _get_library_path(str(target_face_id))
                if not target_face_path:
                    response.status = 400
                    return {"error": "file-not-found"}
            elif stored_target_face_path:
                target_face_path = stored_target_face_path

            if not target_face_path:
                response.status = 400
                return {"error": "missing-params"}

            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code="unsupported-image-format",
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

        if config_id:
            try:
                _ensure_video_task_config_matches(
                    stored_config,
                    input_path,
                    target_face_path=target_face_path,
                    target_face_map=target_face_map,
                    source_map=source_map,
                )
            except RuntimeError as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

        try:
            config_payload = _build_video_task_config_payload(
                input_path=input_path,
                target_face_path=target_face_path,
                target_face_id=str(target_face_id) if target_face_id is not None else None,
                target_faces=target_face_items if has_target_faces else None,
                deep_swap_mode=deep_swap_mode,
                segment_duration_sec=segment_duration_sec,
                segment_overlap_frames=segment_overlap_frames,
                face_sources=face_sources if has_face_sources else None,
                source_map=source_map,
                regions=regions,
                key_frame_ms=key_frame_ms,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {"error": _simplify_task_error(e)}

        active_config_id: Optional[str] = None
        if config_id:
            active_config_id = _store_video_task_config(
                config_payload, config_id=str(config_id)
            )
        elif generate_config_id:
            active_config_id = _store_video_task_config(config_payload)

        if dry_run_config_only:
            if not active_config_id:
                active_config_id = _store_video_task_config(config_payload)
            return {
                "task_id": task_id,
                "status": "config-only",
                "configId": active_config_id,
            }

        _set_video_task_progress(
            task_id,
            status="queued",
            progress=0,
            etaSeconds=None,
            stage="queued",
            frameCount=0,
            totalFrames=0,
            error=None,
            resultFileId=None,
            resultUrl=None,
        )

        _clear_video_task_cancelled(task_id)

            """Handle stage events during video processing."""
        def _on_stage(stage: str):
            if _is_video_task_cancelled(task_id):
                return
            print(f"[INFO] 视频处理阶段: {stage}")
            _set_video_task_progress(
                task_id,
                status="running",
                stage=stage,
                error=None,
            )

            """Handle progress events during video processing."""
        def _on_progress(frame_count: int, total_frames: int, elapsed_seconds: float):
            if _is_video_task_cancelled(task_id):
                return
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
                gpu_provider=gpu_provider,
            )
        elif has_target_faces:
            task_callable = lambda: swap_face_video_deep(
                input_path,
                [item["path"] for item in (target_face_items or [])],
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                segment_duration_sec=segment_duration_sec,
                segment_overlap_frames=segment_overlap_frames,
            )
        else:
            task_callable = lambda: swap_face_video(
                input_path,
                target_face_path,
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
            )

            """Handle completion events after video processing."""
        def _on_completion(res, err):
            if _is_video_task_cancelled(task_id):
                print(f"[WEB] 视频换脸任务已取消，忽略完成回调: task_id={task_id}")
                return

            if res:
                result_id = _register_result(res, [input_path, res])
                _set_video_task_progress(
                    task_id,
                    status="success",
                    progress=100,
                    etaSeconds=0,
                    stage="done",
                    error=None,
                    resultFileId=result_id,
                    resultUrl=f"/api/file/{result_id}",
                )
            else:
                final_error = _simplify_task_error(err)
                _set_video_task_progress(
                    task_id,
                    status="failed",
                    stage="failed",
                    error=final_error,
                    etaSeconds=None,
                )

        try:
            _run_video_task_async(task_id, task_callable, _on_completion)
        except Exception as e:
            _set_video_task_progress(
                task_id,
                status="failed",
                stage="failed",
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
            response.status = 500
            return {"error": _simplify_task_error(e)}
        payload = {"task_id": task_id, "status": "queued"}
        if active_config_id:
            payload["configId"] = active_config_id
        return payload
    except Exception as e:
        print("[ERROR] create_video_task failed:", str(e), "\n", traceback.format_exc())
        if task_id:
            _set_video_task_progress(
                task_id,
                status="failed",
                stage="failed",
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.get("/api/task/video/progress/<task_id>")
    """Return the progress of a video task."""
def get_video_task_progress(task_id):
    response.set_header("Cache-Control", "no-store")
    if not _require_auth():
        return {"error": "unauthorized"}
    return _get_video_task_progress(task_id)


@app.delete("/api/task/<task_id>")
    """Cancel a running task."""
def cancel_task(task_id):
    if not _require_auth():
        return {"error": "unauthorized"}
    AsyncTask.cancel(task_id)
    _mark_video_task_cancelled(task_id)
    _set_video_task_progress(
        task_id,
        status="cancelled",
        stage="cancelled",
        etaSeconds=None,
        error="cancelled",
    )
    return {"success": True}


@app.get("/")
    """Serve the main web page."""
def web_index():
    if os.path.isdir(DIST_DIR):
        return static_file("index.html", root=DIST_DIR)
    response.status = 404
    return "web dist not found"


@app.get("/<filepath:path>")
    """Serve static web assets."""
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