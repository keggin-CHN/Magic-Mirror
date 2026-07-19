import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

from async_tasks import AsyncTask
from fastapi import (
    Body,
    FastAPI,
    File,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from magic.video_task_executor import VideoTaskExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware('http')
async def _request_context_middleware(request_obj: Request, call_next):
    _maybe_run_gc()
    response_obj = await call_next(request_obj)
    response_obj.headers['Referrer-Policy'] = 'no-referrer'
    response_obj.headers['X-Content-Type-Options'] = 'nosniff'
    response_obj.headers['X-Frame-Options'] = 'DENY'
    return response_obj


ALLOWED_IMAGE_EXTS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
    '.tif',
    '.tiff',
    '.json',
}

ALLOWED_VIDEO_EXTS = {
    '.mp4',
    '.mov',
    '.avi',
    '.mkv',
    '.webm',
    '.m4v',
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WEB_DATA_DIR = os.path.abspath(
    os.environ.get('WEB_DATA_DIR') or os.path.join(BASE_DIR, 'data', 'web')
)
UPLOADS_DIR = os.path.abspath(os.path.join(WEB_DATA_DIR, 'uploads'))
LIBRARY_DIR = os.path.abspath(os.path.join(WEB_DATA_DIR, 'library'))
CONFIG_PATH = os.path.join(WEB_DATA_DIR, 'config.json')
DIST_DIR = os.path.abspath(
    os.environ.get('WEB_DIST_DIR') or os.path.join(BASE_DIR, 'dist-web')
)

TOKEN_TTL_SECONDS = 7 * 24 * 3600
AUTH_COOKIE_NAME = 'magic_mirror_token'
PASSWORD_HASH_ITERATIONS = 600_000
LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
LOGIN_LOCKOUT_SECONDS = 60
LOGIN_MAX_FAILURES = 5

# Upload/cache TTL and size limits
UPLOAD_TTL_SECONDS = 24 * 3600
RESULT_TTL_SECONDS = 4 * 3600
PROGRESS_TTL_SECONDS = 6 * 3600
MAX_UPLOAD_BYTES_IMAGE = 50 * 1024 * 1024
MAX_UPLOAD_BYTES_VIDEO = 2 * 1024 * 1024 * 1024
UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024
SAFE_FILENAME_PATTERN = re.compile(r'[^A-Za-z0-9._\-]+')

TOKENS: Dict[str, float] = {}
TOKENS_LOCK = threading.RLock()
LOGIN_FAILURES: Dict[str, List[float]] = {}
LOGIN_FAILURES_LOCK = threading.RLock()
CONFIG_LOCK = threading.RLock()

UPLOADS: Dict[str, Dict[str, object]] = {}
UPLOADS_LOCK = threading.RLock()

RESULTS: Dict[str, Dict[str, object]] = {}
RESULTS_LOCK = threading.RLock()

VIDEO_TASK_PROGRESS: Dict[str, Dict[str, object]] = {}
VIDEO_TASK_PROGRESS_LOCK = threading.RLock()
VIDEO_TASK_CANCELLED = set()
VIDEO_TASK_CANCELLED_LOCK = threading.RLock()
VIDEO_TASK_START_LOCK = threading.RLock()
VIDEO_TASK_EXECUTOR = VideoTaskExecutor(name_prefix='WebVideoTask')
IMAGE_TASKS: Dict[str, threading.Event] = {}

VIDEO_TASK_CONFIGS: Dict[str, Dict[str, object]] = {}
VIDEO_TASK_CONFIGS_LOCK = threading.RLock()
VIDEO_TASK_CONFIG_TTL_SECONDS = 7 * 24 * 3600
VIDEO_TASK_CONFIG_TOKEN_PREFIX = 'cfg1'
DEFAULT_VIDEO_TASK_CONFIG_SECRET = 'magic-mirror-config-secret'
VIDEO_TASK_CONFIG_SECRET = os.environ.get(
    'VIDEO_TASK_CONFIG_SECRET', DEFAULT_VIDEO_TASK_CONFIG_SECRET
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
    with CONFIG_LOCK:
        if not os.path.exists(CONFIG_PATH):
            _save_config(_build_password_config('123456'))
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)


def _init_config_from_environment() -> int:
    """Create the web credential config from WEB_INITIAL_PASSWORD if missing."""
    initial_password = os.environ.get('WEB_INITIAL_PASSWORD', '').strip()
    with CONFIG_LOCK:
        if os.path.exists(CONFIG_PATH):
            print(f'[WEB] config already exists: {CONFIG_PATH}')
            return 0
        if not initial_password:
            print('[WEB] WEB_INITIAL_PASSWORD is required for --init-config')
            return 2
        _save_config(_build_password_config(initial_password))
    print(f'[WEB] initialized credential config: {CONFIG_PATH}')
    return 0


def _save_config(cfg: dict) -> None:
    """Atomically persist the server configuration to disk."""
    config_dir = os.path.dirname(CONFIG_PATH)
    os.makedirs(config_dir, exist_ok=True)
    temp_path = os.path.join(config_dir, f'.config-{uuid.uuid4().hex}.tmp')
    with CONFIG_LOCK:
        try:
            with open(temp_path, 'x', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, CONFIG_PATH)
            try:
                os.chmod(CONFIG_PATH, 0o600)
            except OSError:
                pass
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _build_password_config(password: str) -> dict:
    """Build a salted PBKDF2 password record for on-disk storage."""
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return {
        'passwordHash': password_hash.hex(),
        'passwordSalt': salt.hex(),
        'passwordIterations': PASSWORD_HASH_ITERATIONS,
    }


def _verify_password(password: str, cfg: dict) -> bool:
    """Verify hashed credentials while accepting legacy plaintext configs."""
    legacy_password = cfg.get('password')
    if isinstance(legacy_password, str):
        return hmac.compare_digest(password, legacy_password)

    try:
        iterations = int(cfg['passwordIterations'])
        if iterations <= 0:
            return False
        salt = bytes.fromhex(str(cfg['passwordSalt']))
        expected_hash = bytes.fromhex(str(cfg['passwordHash']))
    except (KeyError, TypeError, ValueError):
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        'sha256', password.encode('utf-8'), salt, iterations
    )
    return hmac.compare_digest(actual_hash, expected_hash)


def _issue_token() -> str:
    """Generate a new authentication token."""
    token = secrets.token_urlsafe(32)
    with TOKENS_LOCK:
        TOKENS[token] = time.time()
    return token


def _cleanup_tokens() -> None:
    """Remove expired authentication tokens."""
    now = time.time()
    with TOKENS_LOCK:
        expired = [
            token
            for token, created in TOKENS.items()
            if now - created > TOKEN_TTL_SECONDS
        ]
        for token in expired:
            TOKENS.pop(token, None)


def _extract_token(request_obj: Request | WebSocket) -> Optional[str]:
    """Extract the authentication token from an HTTP or WebSocket request."""
    auth = request_obj.headers.get('Authorization', '')
    if auth.lower().startswith('bearer '):
        return auth[7:].strip()
    cookie_token = getattr(request_obj, 'cookies', {}).get(AUTH_COOKIE_NAME)
    return (
        cookie_token
        or request_obj.headers.get('X-Token')
        or request_obj.query_params.get('token')
    )


def _set_auth_cookie(response_obj: Response, token: str) -> None:
    """Attach an HttpOnly same-site auth cookie for same-origin web clients."""
    response_obj.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=TOKEN_TTL_SECONDS,
        httponly=True,
        samesite='lax',
    )


def _validate_token(token: Optional[str]) -> bool:
    """Validate a bearer token and refresh its activity timestamp."""
    _cleanup_tokens()
    if not token:
        return False
    with TOKENS_LOCK:
        if token not in TOKENS:
            return False
        TOKENS[token] = time.time()
    return True


def _require_auth(request_obj: Request, response_obj: Response) -> bool:
    """Validate the request token and reject unauthorized calls."""
    if not _validate_token(_extract_token(request_obj)):
        response_obj.status_code = 401
        return False
    return True


def _login_rate_limit_key(request_obj: Request) -> str:
    """Return the client key used for login failure throttling."""
    if request_obj.client and request_obj.client.host:
        return request_obj.client.host
    forwarded_for = request_obj.headers.get('X-Forwarded-For', '')
    if forwarded_for:
        return forwarded_for.split(',', 1)[0].strip() or 'unknown'
    return 'unknown'


def _prune_login_failures(key: str, now: Optional[float] = None) -> List[float]:
    """Drop stale failed login attempts and return active failures."""
    current_time = time.time() if now is None else now
    with LOGIN_FAILURES_LOCK:
        active = [
            failed_at
            for failed_at in LOGIN_FAILURES.get(key, [])
            if current_time - failed_at <= LOGIN_FAILURE_WINDOW_SECONDS
        ]
        if active:
            LOGIN_FAILURES[key] = active
        else:
            LOGIN_FAILURES.pop(key, None)
        return active


def _is_login_rate_limited(key: str, now: Optional[float] = None) -> bool:
    """Return whether the client has too many recent failed logins."""
    current_time = time.time() if now is None else now
    failures = _prune_login_failures(key, current_time)
    return (
        len(failures) >= LOGIN_MAX_FAILURES
        and current_time - failures[-1] <= LOGIN_LOCKOUT_SECONDS
    )


def _record_failed_login(key: str, now: Optional[float] = None) -> None:
    """Record a failed login attempt for throttling."""
    current_time = time.time() if now is None else now
    with LOGIN_FAILURES_LOCK:
        failures = [
            failed_at
            for failed_at in LOGIN_FAILURES.get(key, [])
            if current_time - failed_at <= LOGIN_FAILURE_WINDOW_SECONDS
        ]
        failures.append(current_time)
        LOGIN_FAILURES[key] = failures


def _clear_login_failures(key: str) -> None:
    """Clear failed login attempts after successful authentication."""
    with LOGIN_FAILURES_LOCK:
        LOGIN_FAILURES.pop(key, None)


def _ext(path: str) -> str:
    """Get the file extension from a path."""
    return os.path.splitext(path)[1].lower()


def _validate_file(path: str, allowed_exts: set, *, missing_code: str):
    """Validate a file exists and has an allowed extension."""
    if not path:
        raise RuntimeError('missing-params')
    if not os.path.exists(path):
        raise FileNotFoundError('file-not-found')
    if _ext(path) not in allowed_exts:
        raise RuntimeError(missing_code)


def _simplify_task_error(err: object) -> str:
    """Simplify a task error to a human-readable string."""
    msg = (str(err) if err is not None else '').lower()
    codes = [
        'missing-params',
        'missing-face-sources',
        'invalid-face-source-binding',
        'face-source-not-found',
        'file-not-found',
        'file-too-large',
        'upload-save-failed',
        'invalid-path',
        'unsupported-image-format',
        'unsupported-video-format',
        'unsupported-file-format',
        'image-decode-failed',
        'no-face-detected',
        'no-face-in-selected-regions',
        'swap-failed',
        'video-open-failed',
        'video-write-failed',
        'video-output-missing',
        'audio-mux-failed',
        'video-frame-read-failed',
        'output-write-failed',
        'invalid-regions',
        'config-mismatch',
        'config-not-found',
        'task-already-running',
        'result-path-already-registered',
        'cancelled',
    ]
    for code in codes:
        if code in msg:
            return code
    return 'internal'


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename by removing unsafe characters and limiting length."""
    base = os.path.basename(name or 'upload')
    base = base.replace(' ', '_')
    cleaned = SAFE_FILENAME_PATTERN.sub('_', base).strip('._-') or 'upload'
    if len(cleaned) > 200:
        root, ext = os.path.splitext(cleaned)
        cleaned = root[: 200 - len(ext)] + ext
    return cleaned


def _canonical_path(path: str) -> str:
    """Return a normalized real path for path comparison."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _entry_path(path: str) -> str:
    """Return a normalized path without following the final filesystem entry."""
    return os.path.normcase(os.path.abspath(path))


def _is_path_within(parent: str, child: str) -> bool:
    """Check child path is within parent directory (path traversal defense)."""
    try:
        parent_real = _canonical_path(parent)
        child_real = _canonical_path(child)
        return os.path.commonpath([parent_real, child_real]) == parent_real
    except (ValueError, OSError):
        return False


def _is_immutable_web_asset(path: str) -> bool:
    """Return whether a dist file is a Vite hashed asset."""
    try:
        relative_path = os.path.relpath(path, DIST_DIR).replace('\\', '/')
    except ValueError:
        return False
    if not relative_path.startswith('assets/'):
        return False
    return bool(re.search(r'[-.][A-Za-z0-9_-]{8,}\.[^.]+$', os.path.basename(path)))


def _web_dist_file_response(path: str):
    """Serve a file from the web dist directory with appropriate caching."""
    if _is_immutable_web_asset(path):
        cache_control = 'public, max-age=31536000, immutable'
    else:
        cache_control = 'no-store'
    return FileResponse(path, headers={'Cache-Control': cache_control})


def _web_index_response():
    """Return the SPA index response when the web dist is complete."""
    index_path = os.path.abspath(os.path.join(DIST_DIR, 'index.html'))
    if (
        os.path.isdir(DIST_DIR)
        and _is_path_within(DIST_DIR, index_path)
        and os.path.isfile(index_path)
    ):
        return _web_dist_file_response(index_path)
    return None


def _is_managed_data_path(path: str) -> bool:
    """Check whether a path is inside directories managed by the web server."""
    if not path:
        return False
    managed_roots = {WEB_DATA_DIR, UPLOADS_DIR, LIBRARY_DIR}
    return any(root and _is_path_within(root, path) for root in managed_roots)


def _is_managed_entry_path(path: str) -> bool:
    """Check whether the directory entry itself is under managed data roots."""
    if not path:
        return False
    managed_roots = {WEB_DATA_DIR, UPLOADS_DIR, LIBRARY_DIR}
    try:
        child_path = _entry_path(path)
        for root in managed_roots:
            if not root:
                continue
            root_path = _entry_path(root)
            if os.path.commonpath([root_path, child_path]) == root_path:
                return True
    except (ValueError, OSError):
        return False
    return False


def _save_upload(upload_file, dest_dir: str, *, max_bytes: Optional[int] = None):
    """Save an uploaded file to disk without loading it all into memory."""
    raw = getattr(upload_file, 'file', None)
    if raw is None:
        raise RuntimeError('missing-params')

    filename = _sanitize_filename(getattr(upload_file, 'filename', ''))
    ext = os.path.splitext(filename)[1].lower()
    file_id = uuid.uuid4().hex
    safe_name = f'{file_id}{ext}' if ext else file_id
    save_path = os.path.abspath(os.path.join(dest_dir, safe_name))
    if not _is_path_within(dest_dir, save_path):
        raise RuntimeError('invalid-path')

    os.makedirs(dest_dir, exist_ok=True)
    bytes_written = 0
    try:
        with open(save_path, 'wb') as f:
            try:
                raw.seek(0)
            except (OSError, AttributeError):
                pass
            while True:
                chunk = raw.read(UPLOAD_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if max_bytes is not None and bytes_written > max_bytes:
                    raise RuntimeError('file-too-large')
                f.write(chunk)
    except RuntimeError as error:
        try:
            os.remove(save_path)
        except OSError:
            pass
        if 'file-too-large' in str(error):
            raise
        raise RuntimeError('upload-save-failed') from error
    except Exception as error:
        try:
            os.remove(save_path)
        except OSError:
            pass
        raise RuntimeError('upload-save-failed') from error
    return file_id, save_path, safe_name


def _register_upload(file_id: str, path: str, kind: str) -> None:
    """Register an uploaded file for later retrieval."""
    now = time.time()
    with UPLOADS_LOCK:
        UPLOADS[file_id] = {
            'path': path,
            'kind': kind,
            'createdAt': now,
            'lastUsedAt': now,
            'activeRefs': 0,
        }


def _clone_json_payload(payload):
    """Deep clone a JSON-serializable payload."""
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _build_video_task_config_token(payload: Dict[str, object]) -> str:
    """Build a signed config token for a video task."""
    return build_video_task_config_token(payload, VIDEO_TASK_CONFIG_SECRET)


def _parse_video_task_config_token(config_id: str) -> Optional[Dict[str, object]]:
    """Parse and verify a video task config token."""
    return parse_video_task_config_token(
        str(config_id),
        VIDEO_TASK_CONFIG_SECRET,
        legacy_ttl_seconds=VIDEO_TASK_CONFIG_TTL_SECONDS,
    )


def _register_result(result_path: str) -> str:
    """Register an output file for download and result-owned cleanup."""
    result_id = uuid.uuid4().hex
    now = time.time()
    absolute_path = os.path.abspath(result_path)
    if not _is_managed_data_path(absolute_path):
        raise RuntimeError('invalid-path')
    ownership_key = _canonical_path(absolute_path)
    with RESULTS_LOCK:
        for entry in RESULTS.values():
            existing_path = entry.get('path')
            if (
                isinstance(existing_path, str)
                and _canonical_path(existing_path) == ownership_key
            ):
                raise RuntimeError('result-path-already-registered')
        RESULTS[result_id] = {
            'path': absolute_path,
            # Uploaded inputs have their own lifecycle in UPLOADS. A result must
            # never take ownership of them because the same upload can be reused
            # by another image or video task.
            'delete_paths': [absolute_path],
            'name': os.path.basename(absolute_path),
            'createdAt': now,
            'lastAccessedAt': now,
        }
    return result_id


def _get_upload_path(file_id: str) -> Optional[str]:
    """Get the file path for an uploaded file."""
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        if not item:
            return None
        item['lastUsedAt'] = time.time()
        return item.get('path')


class _UploadPin:
    """Idempotent lease that keeps one uploaded file alive."""

    def __init__(self, file_id: str, path: str):
        self.file_id = file_id
        self.path = path
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        _unpin_upload(self.file_id)


def _pin_upload(file_id: str, expected_kind: Optional[str] = None):
    """Keep an upload alive while a task is queued or running."""
    normalized_id = str(file_id)
    with UPLOADS_LOCK:
        item = UPLOADS.get(normalized_id)
        if not item:
            return None
        if expected_kind and item.get('kind') != expected_kind:
            return None
        item['activeRefs'] = int(item.get('activeRefs', 0)) + 1
        item['lastUsedAt'] = time.time()
        path = item.get('path')
    if not isinstance(path, str):
        _unpin_upload(normalized_id)
        return None
    return _UploadPin(normalized_id, path)


def _unpin_upload(file_id: str) -> None:
    """Release a task reference to an uploaded file."""
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        if not item:
            return
        item['activeRefs'] = max(0, int(item.get('activeRefs', 0)) - 1)
        item['lastUsedAt'] = time.time()


def _get_upload_kind(file_id: str) -> Optional[str]:
    """Get the file kind for an uploaded file."""
    with UPLOADS_LOCK:
        item = UPLOADS.get(file_id)
        return item.get('kind') if item else None


def _get_result_info(file_id: str) -> Optional[Dict[str, object]]:
    """Get info about a registered result."""
    with RESULTS_LOCK:
        info = RESULTS.get(file_id)
        if not info:
            return None
        info['lastAccessedAt'] = time.time()
        return info.copy()


def _safe_delete(path: str) -> None:
    """Safely delete a managed data file, ignoring errors."""
    try:
        absolute_path = os.path.abspath(path)
        if os.path.islink(absolute_path):
            if not _is_managed_entry_path(absolute_path):
                return
            os.remove(absolute_path)
            return
        if _is_managed_data_path(absolute_path) and os.path.isfile(absolute_path):
            os.remove(absolute_path)
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
            if int(entry.get('activeRefs', 0)) <= 0
            and now
            - float(entry.get('lastUsedAt', entry.get('createdAt', now)))
            > UPLOAD_TTL_SECONDS
        ]
        for fid in expired_ids:
            entry = UPLOADS.pop(fid, None)
            if entry:
                p = entry.get('path')
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
            if now
            - float(entry.get('lastAccessedAt', entry.get('createdAt', now)))
            > RESULT_TTL_SECONDS
        ]
        for rid in expired_ids:
            entry = RESULTS.pop(rid, None)
            if entry:
                expired_delete_lists.append(list(entry.get('delete_paths') or []))
    for paths in expired_delete_lists:
        for p in paths:
            _safe_delete(p)


def _cleanup_expired_progress() -> None:
    """Remove finished task progress records older than PROGRESS_TTL_SECONDS."""
    now = time.time()
    finished_states = {'success', 'failed', 'cancelled'}
    with VIDEO_TASK_PROGRESS_LOCK:
        to_remove = []
        for task_id in list(VIDEO_TASK_PROGRESS.keys()):
            state = VIDEO_TASK_PROGRESS.get(task_id) or {}
            status = state.get('status')
            finished_at = state.get('_finishedAt')
            if status in finished_states:
                if finished_at is None:
                    state['_finishedAt'] = now
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
            _canonical_path(str(entry.get('path', '')))
            for entry in UPLOADS.values()
            if entry.get('path')
        }
    try:
        for name in os.listdir(UPLOADS_DIR):
            full = os.path.abspath(os.path.join(UPLOADS_DIR, name))
            is_link = os.path.islink(full)
            if not os.path.isfile(full) and not is_link:
                continue
            if _canonical_path(full) in known_paths:
                continue
            try:
                mtime = os.lstat(full).st_mtime if is_link else os.path.getmtime(full)
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
        print('[WEB] GC failed:', traceback.format_exc())


# ─── End GC ───────────────────────────────────────────────────────────────────


def _invalidate_library_cache() -> None:
    """Invalidate the face library cache."""
    global _LIBRARY_CACHE_MTIME, _LIBRARY_CACHE_ITEMS
    with _LIBRARY_CACHE_LOCK:
        _LIBRARY_CACHE_MTIME = None
        _LIBRARY_CACHE_ITEMS = []


def _list_library_items() -> List[Dict[str, str]]:
    """List all items in the face library."""
    global _LIBRARY_CACHE_MTIME, _LIBRARY_CACHE_ITEMS
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
            if not entry.is_file() or not _is_path_within(LIBRARY_DIR, entry.path):
                continue
            if _ext(entry.name) not in ALLOWED_IMAGE_EXTS:
                continue
            items.append(
                {
                    'id': entry.name,
                    'name': entry.name,
                    'url': f'/api/library/{entry.name}',
                }
            )
    except OSError:
        return []

    with _LIBRARY_CACHE_LOCK:
        _LIBRARY_CACHE_MTIME = dir_mtime
        _LIBRARY_CACHE_ITEMS = [item.copy() for item in items]

    return items


def _get_library_path(item_id: str) -> Optional[str]:
    """Get the file path for a library item."""
    if not item_id:
        return None
    path = os.path.abspath(os.path.join(LIBRARY_DIR, os.path.basename(item_id)))
    if not _is_path_within(LIBRARY_DIR, path) or not os.path.isfile(path):
        return None
    if _ext(path) not in ALLOWED_IMAGE_EXTS:
        return None
    return path


def _set_video_task_progress(task_id: str, **updates):
    """Update the progress of a video task."""
    status = updates.get('status')
    if status in {'success', 'failed', 'cancelled'} and '_finishedAt' not in updates:
        updates['_finishedAt'] = time.time()
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id, {})
        state.update(updates)
        VIDEO_TASK_PROGRESS[task_id] = state


def _get_video_task_progress(task_id: str):
    """Get the current progress of a video task."""
    _maybe_run_gc()
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id)
        if not state:
            return {
                'status': 'idle',
                'progress': 0,
                'etaSeconds': None,
                'stage': None,
            }
        public_state = state.copy()
    public_state.pop('_finishedAt', None)
    return public_state


def _mark_video_task_cancelled(task_id: str):
    """Mark a video task as cancelled."""
    with VIDEO_TASK_CANCELLED_LOCK:
        VIDEO_TASK_CANCELLED.add(task_id)


def _clear_video_task_cancelled(task_id: str):
    """Clear the cancelled status of a video task."""
    with VIDEO_TASK_CANCELLED_LOCK:
        VIDEO_TASK_CANCELLED.discard(task_id)


def _is_video_task_cancelled(task_id: str) -> bool:
    """Check if a video task has been cancelled."""
    with VIDEO_TASK_CANCELLED_LOCK:
        return task_id in VIDEO_TASK_CANCELLED


def _run_image_task_with_upload_pin(input_id, task_id, task_callable):
    """Run an image task while preventing its uploaded input from expiring."""
    normalized_task_id = str(task_id)
    cancel_event = threading.Event()
    _cleanup_expired_progress()
    with VIDEO_TASK_START_LOCK:
        if (
            normalized_task_id in IMAGE_TASKS
            or VIDEO_TASK_EXECUTOR.is_active(normalized_task_id)
            or normalized_task_id in VIDEO_TASK_PROGRESS
        ):
            return None, RuntimeError('task-already-running')
        IMAGE_TASKS[normalized_task_id] = cancel_event

    upload_pin = _pin_upload(str(input_id), expected_kind='image')
    if not upload_pin:
        with VIDEO_TASK_START_LOCK:
            if IMAGE_TASKS.get(normalized_task_id) is cancel_event:
                IMAGE_TASKS.pop(normalized_task_id, None)
        return None, FileNotFoundError('file-not-found')

    def _guarded_task():
        if cancel_event.is_set():
            raise RuntimeError('cancelled')
        return task_callable()

    try:
        return AsyncTask.run(_guarded_task, task_id=normalized_task_id)
    finally:
        upload_pin.release()
        with VIDEO_TASK_START_LOCK:
            if IMAGE_TASKS.get(normalized_task_id) is cancel_event:
                IMAGE_TASKS.pop(normalized_task_id, None)


def _run_video_task_async(
    task_id: str,
    task_callable,
    on_completion,
    upload_pins=None,
):
    """Queue a video task in the bounded background executor."""
    active_upload_pins = list(upload_pins or [])

    def _completion(res, err):
        try:
            on_completion(res, err)
        except Exception:
            print(
                '[ERROR] web video task completion callback failed:\n',
                traceback.format_exc(),
            )
        finally:
            _clear_video_task_cancelled(task_id)
            for upload_pin in active_upload_pins:
                upload_pin.release()

    try:
        VIDEO_TASK_EXECUTOR.submit(
            task_id,
            task_callable,
            _completion,
            on_cancel_result=lambda result: _safe_delete(str(result)),
        )
    except Exception:
        for upload_pin in active_upload_pins:
            upload_pin.release()
        raise


def _cleanup_video_task_configs() -> None:
    """Clean up expired video task configurations."""
    now = time.time()
    with VIDEO_TASK_CONFIGS_LOCK:
        expired = [
            config_id
            for config_id, item in VIDEO_TASK_CONFIGS.items()
            if now - float(item.get('createdAt', 0)) > VIDEO_TASK_CONFIG_TTL_SECONDS
        ]
        for config_id in expired:
            VIDEO_TASK_CONFIGS.pop(config_id, None)


def _store_video_task_config(
    payload: Dict[str, object], config_id: Optional[str] = None
) -> str:
    """Store a video task configuration."""
    _cleanup_video_task_configs()
    next_id = str(config_id or _build_video_task_config_token(payload))
    with VIDEO_TASK_CONFIGS_LOCK:
        VIDEO_TASK_CONFIGS[next_id] = {
            'createdAt': time.time(),
            'config': _clone_json_payload(payload),
        }
    return next_id


def _get_video_task_config(config_id: str) -> Optional[Dict[str, object]]:
    """Get a stored video task configuration."""
    if not config_id:
        return None
    _cleanup_video_task_configs()
    with VIDEO_TASK_CONFIGS_LOCK:
        item = VIDEO_TASK_CONFIGS.get(str(config_id))
        if item:
            item['createdAt'] = time.time()
            config = item.get('config')
            if isinstance(config, dict):
                return _clone_json_payload(config)

    return _parse_video_task_config_token(str(config_id))


def _extract_stored_path(file_entry):
    """Extract the file path from a stored entry."""
    if isinstance(file_entry, str) and file_entry:
        return file_entry
    if isinstance(file_entry, dict):
        path = file_entry.get('path')
        if isinstance(path, str) and path:
            return path
    return None


def _extract_stored_face_sources(face_sources):
    """Extract face source paths from stored entries."""
    if not isinstance(face_sources, list):
        return None

    resolved = []
    for source in face_sources:
        if not isinstance(source, dict):
            return None
        source_id = source.get('id')
        source_path = _extract_stored_path(source)
        if source_id is None or not source_path:
            return None
        resolved.append({'id': str(source_id), 'path': source_path})

    return resolved or None


def _resolve_face_reference_path(face_ref: str) -> Optional[str]:
    """Resolve a face reference to a file path."""
    if not face_ref:
        return None
    library_path = _get_library_path(str(face_ref))
    if library_path:
        return library_path
    if os.path.exists(face_ref):
        return face_ref
    return None


def _resolve_target_face_items(target_faces) -> List[Dict[str, str]]:
    """Resolve target face items to file paths."""
    if not isinstance(target_faces, list) or len(target_faces) == 0:
        raise RuntimeError('missing-params')

    resolved: List[Dict[str, str]] = []
    for index, target in enumerate(target_faces):
        target_id = f'target-{index + 1}'
        target_ref = None

        if isinstance(target, str):
            target_ref = target
        elif isinstance(target, dict):
            target_id = str(target.get('id') or target_id)
            target_ref = target.get('path') or target.get('id')
        else:
            raise RuntimeError('missing-params')

        if not target_ref:
            raise RuntimeError('missing-params')

        target_path = _resolve_face_reference_path(str(target_ref))
        if not target_path:
            raise FileNotFoundError('file-not-found')

        _validate_file(
            target_path,
            ALLOWED_IMAGE_EXTS,
            missing_code='unsupported-image-format',
        )
        resolved.append({'id': str(target_id), 'path': target_path})

    return resolved


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
    gpu_provider: str = 'auto',
) -> Dict[str, object]:
    """Build the config payload for a video task."""
    payload: Dict[str, object] = {
        'inputVideo': {
            'path': input_path,
            'sha256': compute_file_sha256(input_path),
        },
        'regions': regions if isinstance(regions, list) else None,
        'keyFrameMs': max(0, int(key_frame_ms or 0)),
        'useGpu': bool(use_gpu),
        'gpuProvider': str(gpu_provider or 'auto').strip().lower(),
    }

    if isinstance(face_sources, list) and len(face_sources) > 0:
        normalized_sources = []
        source_map = source_map or {}
        for source in face_sources:
            if not isinstance(source, dict):
                raise RuntimeError('missing-face-sources')
            source_id = source.get('id')
            if source_id is None:
                raise RuntimeError('missing-face-sources')
            source_path = source_map.get(str(source_id))
            if not source_path:
                raise RuntimeError('missing-face-sources')
            normalized_sources.append(
                {
                    'id': str(source_id),
                    'path': source_path,
                    'sha256': compute_file_sha256(source_path),
                }
            )
        payload['faceSources'] = normalized_sources
    elif isinstance(target_faces, list) and len(target_faces) > 0:
        normalized_targets = []
        for idx, target in enumerate(target_faces):
            if not isinstance(target, dict):
                raise RuntimeError('missing-params')
            target_id = target.get('id')
            target_path = target.get('path')
            if not target_path:
                raise RuntimeError('missing-params')
            normalized_targets.append(
                {
                    'id': str(target_id or f'target-{idx + 1}'),
                    'path': target_path,
                    'sha256': compute_file_sha256(target_path),
                }
            )
        payload['targetFaces'] = normalized_targets
        payload['deepSwapMode'] = bool(deep_swap_mode)
        payload['segmentDurationSec'] = max(1, int(segment_duration_sec or 1))
        payload['segmentOverlapFrames'] = max(0, int(segment_overlap_frames or 0))
    else:
        if not target_face_path:
            raise RuntimeError('missing-params')
        if target_face_id is not None:
            payload['targetFaceId'] = str(target_face_id)
        payload['targetFace'] = {
            'path': target_face_path,
            'sha256': compute_file_sha256(target_face_path),
        }

    return payload


def _ensure_video_task_config_matches(
    config: Dict[str, object],
    input_path: str,
    *,
    target_face_path: Optional[str] = None,
    target_face_map: Optional[Dict[str, str]] = None,
    source_map: Optional[Dict[str, str]] = None,
) -> None:
    """Ensure a video task config matches expectations."""
    expected_input_sha256 = get_expected_input_video_sha256(config)
    if expected_input_sha256 and not verify_file_sha256(
        input_path, expected_input_sha256
    ):
        raise RuntimeError('config-mismatch')

    expected_target_sha256 = get_expected_target_face_sha256(config)
    if expected_target_sha256:
        if not target_face_path or not verify_file_sha256(
            target_face_path, expected_target_sha256
        ):
            raise RuntimeError('config-mismatch')

    expected_target_face_sha256_map = get_expected_target_faces_sha256_map(config)
    if expected_target_face_sha256_map:
        if not isinstance(target_face_map, dict):
            raise RuntimeError('config-mismatch')
        for target_id, expected_sha256 in expected_target_face_sha256_map.items():
            target_path = target_face_map.get(str(target_id))
            if not target_path or not verify_file_sha256(target_path, expected_sha256):
                raise RuntimeError('config-mismatch')

    expected_source_sha256_map = get_expected_face_source_sha256_map(config)
    if expected_source_sha256_map:
        if not isinstance(source_map, dict):
            raise RuntimeError('config-mismatch')
        for source_id, expected_sha256 in expected_source_sha256_map.items():
            source_path = source_map.get(str(source_id))
            if not source_path or not verify_file_sha256(source_path, expected_sha256):
                raise RuntimeError('config-mismatch')


def _cleanup_result(result_id: str, delete_paths: List[str]) -> None:
    """Clean up a result and files owned by that result."""
    with RESULTS_LOCK:
        RESULTS.pop(result_id, None)
    for path in delete_paths:
        _safe_delete(path)


@app.options('/api/{path:path}')
def handle_options(path: str):
    """Handle CORS preflight OPTIONS requests."""
    return {}


@app.get('/api/status')
def status(response: Response, request: Request):
    """Return the server status."""
    return {'status': 'running'}


@app.post('/api/prepare')
def prepare(response: Response, request: Request):
    """Prepare a file for face detection."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    return {'success': load_models()}


@app.post('/api/login')
def login(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Authenticate and return a token."""
    rate_limit_key = _login_rate_limit_key(request)
    if _is_login_rate_limited(rate_limit_key):
        response.status_code = 429
        response.headers['Retry-After'] = str(LOGIN_LOCKOUT_SECONDS)
        return {'error': 'too-many-login-attempts'}

    password = str(body.get('password', '')).strip()
    cfg = _load_config()
    if not _verify_password(password, cfg):
        _record_failed_login(rate_limit_key)
        response.status_code = 401
        return {'error': 'invalid-credential'}
    if 'password' in cfg:
        _save_config(_build_password_config(password))
    _clear_login_failures(rate_limit_key)
    token = _issue_token()
    _set_auth_cookie(response, token)
    return {'token': token}


@app.post('/api/credential')
def update_credential(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Update the server password."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}

    new_password = str(body.get('password', '')).strip()
    if not new_password:
        response.status_code = 400
        return {'error': 'missing-params'}
    _save_config(_build_password_config(new_password))
    with TOKENS_LOCK:
        TOKENS.clear()
    token = _issue_token()
    _set_auth_cookie(response, token)
    return {'success': True, 'token': token}


@app.get('/api/library')
def list_library(response: Response, request: Request):
    """List all items in the face library."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    return {'items': _list_library_items()}


@app.post('/api/library/upload')
def upload_library(response: Response, request: Request, file: UploadFile = File(...)):
    """Upload a new face to the library."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    upload_file = file
    if not upload_file:
        response.status_code = 400
        return {'error': 'missing-params'}
    ext = _ext(upload_file.filename or '')
    if ext not in ALLOWED_IMAGE_EXTS:
        response.status_code = 400
        return {'error': 'unsupported-image-format'}
    try:
        _, save_path, safe_name = _save_upload(
            upload_file, LIBRARY_DIR, max_bytes=MAX_UPLOAD_BYTES_IMAGE
        )
    except RuntimeError as e:
        response.status_code = 400
        return {'error': _simplify_task_error(e)}
    _invalidate_library_cache()
    return {
        'id': safe_name,
        'name': safe_name,
        'url': f'/api/library/{safe_name}',
    }


@app.get('/api/library/{filename}')
def get_library_file(response: Response, request: Request, filename):
    """Retrieve a face library file."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    safe_name = os.path.basename(filename or '')
    path = os.path.abspath(os.path.join(LIBRARY_DIR, safe_name))
    if not _is_path_within(LIBRARY_DIR, path) or not os.path.exists(path):
        response.status_code = 404
        return {'error': 'file-not-found'}
    if _ext(path) not in ALLOWED_IMAGE_EXTS:
        response.status_code = 404
        return {'error': 'file-not-found'}
    return FileResponse(path, headers={'Cache-Control': 'no-store'})


@app.post('/api/upload')
def upload_file(response: Response, request: Request, file: UploadFile = File(...)):
    """Upload a file for processing."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    upload_file = file
    if not upload_file:
        response.status_code = 400
        return {'error': 'missing-params'}
    ext = _ext(upload_file.filename or '')
    if ext in ALLOWED_IMAGE_EXTS:
        kind = 'image'
        max_bytes = MAX_UPLOAD_BYTES_IMAGE
    elif ext in ALLOWED_VIDEO_EXTS:
        kind = 'video'
        max_bytes = MAX_UPLOAD_BYTES_VIDEO
    else:
        response.status_code = 400
        return {'error': 'unsupported-file-format'}
    try:
        file_id, save_path, _ = _save_upload(
            upload_file, UPLOADS_DIR, max_bytes=max_bytes
        )
    except RuntimeError as e:
        response.status_code = 400
        return {'error': _simplify_task_error(e)}
    _register_upload(file_id, save_path, kind)
    return {
        'fileId': file_id,
        'url': f'/api/file/{file_id}',
        'type': kind,
        'name': upload_file.filename or 'upload',
    }


@app.get('/api/file/{file_id}')
def get_file(response: Response, request: Request, file_id):
    """Retrieve an uploaded file."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    path = _get_upload_path(file_id)
    if not path:
        info = _get_result_info(file_id)
        if info:
            path = info.get('path')
    if not path or not os.path.exists(path):
        response.status_code = 404
        return {'error': 'file-not-found'}
    abs_path = os.path.abspath(str(path))
    if not (
        _is_path_within(UPLOADS_DIR, abs_path)
        or _is_path_within(LIBRARY_DIR, abs_path)
        or _is_path_within(WEB_DATA_DIR, abs_path)
    ):
        response.status_code = 404
        return {'error': 'file-not-found'}
    return FileResponse(abs_path, headers={'Cache-Control': 'no-store'})


@app.head('/api/download/{file_id}')
@app.get('/api/download/{file_id}')
def download_file(response: Response, request: Request, file_id):
    """Download a result file."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    info = _get_result_info(file_id)
    if not info:
        response.status_code = 404
        return {'error': 'file-not-found'}
    path = info.get('path')
    if not path or not os.path.exists(path):
        response.status_code = 404
        return {'error': 'file-not-found'}
    filename = info.get('name') or os.path.basename(path)
    return FileResponse(
        path,
        headers={'Cache-Control': 'no-store'},
        media_type=mimetypes.guess_type(path)[0] or 'application/octet-stream',
        filename=filename,
    )


@app.post('/api/task/detect-faces')
def detect_faces_for_image(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Detect faces in an uploaded image."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    input_pin = None
    try:

        input_id = body.get('inputFileId')
        regions = body.get('regions')
        input_pin = _pin_upload(str(input_id), expected_kind='image')
        if not input_pin:
            response.status_code = 400
            return {'error': 'file-not-found'}
        input_path = input_pin.path
        _validate_file(
            input_path,
            ALLOWED_IMAGE_EXTS,
            missing_code='unsupported-image-format',
        )
        if regions is not None and not isinstance(regions, list):
            response.status_code = 400
            return {'error': 'missing-params'}
        result = detect_face_boxes_in_image(input_path, regions=regions)
        return {'regions': result}
    except Exception as e:
        response.status_code = 500
        return {'error': _simplify_task_error(e)}
    finally:
        if input_pin is not None:
            input_pin.release()


@app.post('/api/task/video/detect-faces')
def detect_faces_for_video(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Detect faces in a video file."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    input_pin = None
    try:

        input_id = body.get('inputFileId')
        key_frame_ms = body.get('keyFrameMs', 0)
        regions = body.get('regions')
        input_pin = _pin_upload(str(input_id), expected_kind='video')
        if not input_pin:
            response.status_code = 400
            return {'error': 'file-not-found'}
        input_path = input_pin.path
        _validate_file(
            input_path,
            ALLOWED_VIDEO_EXTS,
            missing_code='unsupported-video-format',
        )
        if regions is not None and not isinstance(regions, list):
            response.status_code = 400
            return {'error': 'missing-params'}
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
        response.status_code = 500
        return {'error': _simplify_task_error(e)}
    finally:
        if input_pin is not None:
            input_pin.release()


@app.post('/api/task')
def create_task(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Create a new face swap task."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    input_pin = None
    try:

        raw_task_id = body.get('id')
        task_id = str(raw_task_id).strip() if raw_task_id is not None else ''
        input_id = body.get('inputFileId')
        regions = body.get('regions')
        target_face_id = body.get('targetFaceId')
        face_sources = body.get('faceSources')
        target_faces = body.get('targetFaces')
        deep_swap_mode = bool(body.get('deepSwapMode', False))
        has_face_sources = 'faceSources' in body
        has_target_faces = 'targetFaces' in body or deep_swap_mode

        if not all([task_id, input_id]):
            response.status_code = 400
            return {'error': 'missing-params'}
        input_pin = _pin_upload(str(input_id), expected_kind='image')
        if not input_pin:
            response.status_code = 400
            return {'error': 'file-not-found'}
        input_path = input_pin.path
        try:
            _validate_file(
                input_path,
                ALLOWED_IMAGE_EXTS,
                missing_code='unsupported-image-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        if has_face_sources and has_target_faces:
            response.status_code = 400
            return {'error': 'missing-params'}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status_code = 400
                return {'error': 'missing-face-sources'}
            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                source_id = source.get('id')
                source_path = _get_library_path(str(source_id))
                if not source_id or not source_path:
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                source_map[str(source_id)] = source_path
            if regions:
                if not isinstance(regions, list):
                    response.status_code = 400
                    return {'error': 'invalid-face-source-binding'}
                for region in regions:
                    if not isinstance(region, dict):
                        response.status_code = 400
                        return {'error': 'invalid-face-source-binding'}
                    source_id = region.get('faceSourceId')
                    if not source_id or str(source_id) not in source_map:
                        response.status_code = 400
                        return {'error': 'invalid-face-source-binding'}
                res, err = _run_image_task_with_upload_pin(
                    input_id,
                    task_id,
                    lambda: swap_face_regions_by_sources(
                        input_path, source_map, regions
                    ),
                )
            else:
                fallback_face = next(iter(source_map.values()))
                res, err = _run_image_task_with_upload_pin(
                    input_id,
                    task_id,
                    lambda: swap_face(input_path, fallback_face),
                )
        elif has_target_faces:
            try:
                target_face_items = _resolve_target_face_items(target_faces)
            except (RuntimeError, FileNotFoundError) as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

            res, err = _run_image_task_with_upload_pin(
                input_id,
                task_id,
                lambda: swap_face_deep(
                    input_path,
                    [item['path'] for item in target_face_items],
                    regions=regions,
                ),
            )
        else:
            if not target_face_id:
                response.status_code = 400
                return {'error': 'missing-params'}
            target_face_path = _get_library_path(str(target_face_id))
            if not target_face_path:
                response.status_code = 400
                return {'error': 'file-not-found'}
            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code='unsupported-image-format',
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}
            if regions:
                res, err = _run_image_task_with_upload_pin(
                    input_id,
                    task_id,
                    lambda: swap_face_regions(input_path, target_face_path, regions),
                )
            else:
                res, err = _run_image_task_with_upload_pin(
                    input_id,
                    task_id,
                    lambda: swap_face(input_path, target_face_path),
                )
        if res:
            result_id = _register_result(res)
            return {'resultFileId': result_id, 'resultUrl': f'/api/file/{result_id}'}
        error_code = _simplify_task_error(err)
        response.status_code = 409 if error_code == 'task-already-running' else 500
        return {'error': error_code}
    except Exception as e:
        print('[ERROR] create_task failed:', str(e), '\n', traceback.format_exc())
        response.status_code = 500
        return {'error': _simplify_task_error(e)}
    finally:
        if input_pin is not None:
            input_pin.release()


@app.get('/api/task/video/gpu-modes')
def get_video_gpu_modes(response: Response, request: Request):
    """Return available GPU acceleration modes."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}

    try:
        return get_gpu_acceleration_modes()
    except Exception as e:
        response.status_code = 500
        return {
            'modes': [{'id': 'cpu', 'name': 'CPU'}],
            'availableProviders': [],
            'error': _simplify_task_error(e),
        }


@app.post('/api/task/video')
def create_video_task(
    response: Response,
    request: Request,
    body: dict = Body(default_factory=dict),
):
    """Create a new video face swap task."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    task_id = None
    input_pin = None
    try:

        raw_task_id = body.get('id')
        task_id = str(raw_task_id).strip() if raw_task_id is not None else ''
        input_id = body.get('inputFileId')
        target_face_id = body.get('targetFaceId')
        target_faces = body.get('targetFaces')
        regions = body.get('regions')
        face_sources = body.get('faceSources')
        deep_swap_mode = bool(body.get('deepSwapMode', False))
        segment_duration_sec = body.get('segmentDurationSec', 12)
        segment_overlap_frames = body.get('segmentOverlapFrames', 6)
        key_frame_ms = body.get('keyFrameMs', 0)
        use_gpu = body.get('useGpu', False)
        gpu_provider = str(body.get('gpuProvider', 'auto') or 'auto').strip().lower()
        config_id = body.get('configId')
        generate_config_id = bool(body.get('generateConfigId', False))
        dry_run_config_only = bool(body.get('dryRunConfigOnly', False))

        stored_config = None
        stored_target_face_path = None
        if config_id:
            stored_config = _get_video_task_config(str(config_id))
            if not isinstance(stored_config, dict):
                response.status_code = 400
                return {'error': 'config-not-found'}

            stored_target_face_path = _extract_stored_path(
                stored_config.get('targetFace')
            )

            if target_face_id is None:
                target_face_id = stored_config.get('targetFaceId')
            if target_faces is None and 'targetFaces' in stored_config:
                target_faces = _extract_stored_face_sources(
                    stored_config.get('targetFaces')
                )
            if face_sources is None and 'faceSources' in stored_config:
                face_sources = stored_config.get('faceSources')
            if regions is None and 'regions' in stored_config:
                regions = stored_config.get('regions')
            if 'deepSwapMode' not in body:
                deep_swap_mode = bool(stored_config.get('deepSwapMode', deep_swap_mode))
            if 'segmentDurationSec' not in body:
                segment_duration_sec = stored_config.get(
                    'segmentDurationSec', segment_duration_sec
                )
            if 'segmentOverlapFrames' not in body:
                segment_overlap_frames = stored_config.get(
                    'segmentOverlapFrames', segment_overlap_frames
                )
            if 'keyFrameMs' not in body:
                key_frame_ms = stored_config.get('keyFrameMs', key_frame_ms)
            if 'useGpu' not in body:
                use_gpu = bool(stored_config.get('useGpu', use_gpu))
            if 'gpuProvider' not in body:
                gpu_provider = (
                    str(stored_config.get('gpuProvider', gpu_provider) or gpu_provider)
                    .strip()
                    .lower()
                )

        if gpu_provider in ('dml', 'directml'):
            gpu_provider = 'directml'
        elif gpu_provider == 'cuda':
            gpu_provider = 'cuda'
        elif gpu_provider == 'cpu':
            gpu_provider = 'cpu'
        else:
            gpu_provider = 'auto'

        if gpu_provider == 'cpu':
            use_gpu = False
        elif gpu_provider in ('directml', 'cuda'):
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
            response.status_code = 400
            return {'error': 'missing-params'}
        input_pin = _pin_upload(str(input_id), expected_kind='video')
        if not input_pin:
            response.status_code = 400
            return {'error': 'file-not-found'}
        input_path = input_pin.path
        try:
            _validate_file(
                input_path,
                ALLOWED_VIDEO_EXTS,
                missing_code='unsupported-video-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        source_map: Optional[Dict[str, str]] = None
        target_face_path: Optional[str] = None
        target_face_map: Optional[Dict[str, str]] = None
        target_face_items: Optional[List[Dict[str, str]]] = None

        if has_face_sources and has_target_faces:
            response.status_code = 400
            return {'error': 'missing-params'}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status_code = 400
                return {'error': 'missing-face-sources'}
            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                source_id = source.get('id')
                source_path = _get_library_path(str(source_id))
                if not source_id or not source_path:
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                source_map[str(source_id)] = source_path
            if not isinstance(regions, list) or len(regions) == 0:
                response.status_code = 400
                return {'error': 'invalid-face-source-binding'}
            for region in regions:
                if not isinstance(region, dict):
                    response.status_code = 400
                    return {'error': 'invalid-face-source-binding'}
                source_id = region.get('faceSourceId')
                if not source_id or str(source_id) not in source_map:
                    response.status_code = 400
                    return {'error': 'invalid-face-source-binding'}
        elif has_target_faces:
            try:
                target_face_items = _resolve_target_face_items(target_faces)
            except (RuntimeError, FileNotFoundError) as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

            target_face_map = {
                str(item.get('id')): str(item.get('path'))
                for item in (target_face_items or [])
            }
        else:
            if target_face_id is not None:
                target_face_path = _get_library_path(str(target_face_id))
                if not target_face_path:
                    response.status_code = 400
                    return {'error': 'file-not-found'}
            elif stored_target_face_path:
                target_face_path = stored_target_face_path

            if not target_face_path:
                response.status_code = 400
                return {'error': 'missing-params'}

            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code='unsupported-image-format',
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

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
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

        try:
            config_payload = _build_video_task_config_payload(
                input_path=input_path,
                target_face_path=target_face_path,
                target_face_id=str(target_face_id)
                if target_face_id is not None
                else None,
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
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

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
                'task_id': task_id,
                'status': 'config-only',
                'configId': active_config_id,
            }

        def _on_stage(stage: str):
            """Handle stage events during video processing."""
            if _is_video_task_cancelled(task_id):
                return
            print(f'[INFO] 视频处理阶段: {stage}')
            _set_video_task_progress(
                task_id,
                status='running',
                stage=stage,
                error=None,
            )

        def _on_progress(frame_count: int, total_frames: int, elapsed_seconds: float):
            """Handle progress events during video processing."""
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
                status='running',
                progress=round(progress, 2),
                etaSeconds=eta_seconds,
                frameCount=frame_count,
                totalFrames=total_frames,
                error=None,
            )

        if has_face_sources:
            task_callable = lambda cancel_event: swap_face_video_by_sources(
                input_path,
                source_map,
                regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                cancel_event=cancel_event,
            )
        elif has_target_faces:
            task_callable = lambda cancel_event: swap_face_video_deep(
                input_path,
                [item['path'] for item in (target_face_items or [])],
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                segment_duration_sec=segment_duration_sec,
                segment_overlap_frames=segment_overlap_frames,
                cancel_event=cancel_event,
            )
        else:
            task_callable = lambda cancel_event: swap_face_video(
                input_path,
                target_face_path,
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                cancel_event=cancel_event,
            )

        def _on_completion(res, err):
            """Handle completion events after video processing."""
            if _is_video_task_cancelled(task_id):
                print(f'[WEB] 视频换脸任务已取消，忽略完成回调: task_id={task_id}')
                return

            if res:
                result_id = _register_result(res)
                _set_video_task_progress(
                    task_id,
                    status='success',
                    progress=100,
                    etaSeconds=0,
                    stage='done',
                    error=None,
                    resultFileId=result_id,
                    resultUrl=f'/api/file/{result_id}',
                )
            else:
                final_error = _simplify_task_error(err)
                _set_video_task_progress(
                    task_id,
                    status='failed',
                    stage='failed',
                    error=final_error,
                    etaSeconds=None,
                )

        _cleanup_expired_progress()
        try:
            with VIDEO_TASK_START_LOCK:
                if (
                    VIDEO_TASK_EXECUTOR.is_active(task_id)
                    or task_id in IMAGE_TASKS
                    or task_id in VIDEO_TASK_PROGRESS
                ):
                    response.status_code = 409
                    return {'error': 'task-already-running'}
                _set_video_task_progress(
                    task_id,
                    status='queued',
                    progress=0,
                    etaSeconds=None,
                    stage='queued',
                    frameCount=0,
                    totalFrames=0,
                    error=None,
                    resultFileId=None,
                    resultUrl=None,
                )
                _clear_video_task_cancelled(task_id)
                _run_video_task_async(
                    task_id,
                    task_callable,
                    _on_completion,
                    upload_pins=[input_pin],
                )
            input_pin = None
        except Exception as e:
            _set_video_task_progress(
                task_id,
                status='failed',
                stage='failed',
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
            response.status_code = 500
            return {'error': _simplify_task_error(e)}
        payload = {'task_id': task_id, 'status': 'queued'}
        if active_config_id:
            payload['configId'] = active_config_id
        return payload
    except Exception as e:
        print('[ERROR] create_video_task failed:', str(e), '\n', traceback.format_exc())
        if task_id:
            _set_video_task_progress(
                task_id,
                status='failed',
                stage='failed',
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
        response.status_code = 500
        return {'error': _simplify_task_error(e)}
    finally:
        if input_pin is not None:
            input_pin.release()


@app.get('/api/task/video/progress/{task_id}')
def get_video_task_progress(response: Response, request: Request, task_id):
    """Return the progress of a video task."""
    response.headers.__setitem__('Cache-Control', 'no-store')
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    return _get_video_task_progress(task_id)


@app.delete('/api/task/{task_id}')
def cancel_task(response: Response, request: Request, task_id):
    """Cancel a running task."""
    if not _require_auth(request, response):
        return {'error': 'unauthorized'}
    with VIDEO_TASK_START_LOCK:
        executor_cancelled = VIDEO_TASK_EXECUTOR.cancel(task_id)
        image_cancel_event = IMAGE_TASKS.get(str(task_id))
        image_task_active = image_cancel_event is not None
        if image_cancel_event is not None:
            image_cancel_event.set()
        AsyncTask.cancel(task_id)
        if not executor_cancelled and not image_task_active:
            return {'success': False}
        if executor_cancelled:
            _mark_video_task_cancelled(task_id)
            _set_video_task_progress(
                task_id,
                status='cancelled',
                stage='cancelled',
                etaSeconds=None,
                error='cancelled',
            )
    return {'success': True}


@app.websocket('/api/task/video/ws/{task_id}')
async def web_video_task_ws(websocket: WebSocket, task_id: str):
    if not _validate_token(_extract_token(websocket)):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            progress = _get_video_task_progress(task_id)
            await websocket.send_json(progress)
            if progress.get('status') in {'success', 'failed', 'cancelled'}:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@app.get('/')
def web_index(response: Response, request: Request):
    """Serve the main web page."""
    index_response = _web_index_response()
    if index_response is not None:
        return index_response
    response.status_code = 404
    return 'web dist not found'


@app.get('/{filepath:path}')
def web_assets(response: Response, request: Request, filepath: str):
    """Serve static web assets."""
    if filepath == 'api' or filepath.startswith('api/'):
        response.status_code = 404
        return {'error': 'not-found'}
    if os.path.isdir(DIST_DIR):
        candidate = os.path.abspath(os.path.join(DIST_DIR, filepath))
        if (
            _is_path_within(DIST_DIR, candidate)
            and os.path.exists(candidate)
            and os.path.isfile(candidate)
        ):
            return _web_dist_file_response(candidate)
        index_response = _web_index_response()
        if index_response is not None:
            return index_response
    response.status_code = 404
    return 'web dist not found'


def _parse_env_port(env_name: str, default: int) -> int:
    raw_value = os.environ.get(env_name, str(default))
    try:
        port = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{env_name} must be an integer, got {raw_value!r}'
        ) from error
    if not 1 <= port <= 65535:
        raise ValueError(f'{env_name} must be between 1 and 65535, got {port}')
    return port


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv if argv is None else argv
    if len(args) > 1 and args[1] == '--init-config':
        return _init_config_from_environment()
    try:
        import uvicorn

        host = os.environ.get('WEB_HOST', '0.0.0.0')
        port = _parse_env_port('WEB_PORT', 8033)
        if VIDEO_TASK_CONFIG_SECRET == DEFAULT_VIDEO_TASK_CONFIG_SECRET:
            print(
                '[WEB] warning: VIDEO_TASK_CONFIG_SECRET uses the development '
                'default; set a random value for production deployments'
            )
        print(f'[WEB] starting server on {host}:{port}')
        uvicorn.run(app, host=host, port=port, access_log=False)
    except Exception as error:
        print(f'[WEB] server failed: {error!r}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
