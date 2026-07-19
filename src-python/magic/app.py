import asyncio
import json
import os
import threading
import time
import traceback

from async_tasks import AsyncTask
from fastapi import Body, FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .face import (
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
from .task_config import (
    build_video_task_config_token,
    compute_file_sha256,
    get_expected_face_source_sha256_map,
    get_expected_input_video_sha256,
    get_expected_target_face_sha256,
    get_expected_target_faces_sha256_map,
    parse_video_task_config_token,
    verify_file_sha256,
)
from .video_task_executor import VideoTaskExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


VIDEO_TASK_PROGRESS = {}
VIDEO_TASK_PROGRESS_LOCK = threading.RLock()
VIDEO_TASK_CANCELLED = set()
VIDEO_TASK_CANCELLED_LOCK = threading.RLock()
VIDEO_TASK_START_LOCK = threading.RLock()
VIDEO_TASK_EXECUTOR = VideoTaskExecutor(name_prefix='DesktopVideoTask')
IMAGE_TASKS = {}

# GC constants for progress records
_PROGRESS_TTL_SECONDS = 6 * 3600
_LAST_PROGRESS_GC_AT = 0.0
_PROGRESS_GC_LOCK = threading.RLock()
_PROGRESS_GC_INTERVAL = 5 * 60

VIDEO_TASK_CONFIGS = {}
VIDEO_TASK_CONFIGS_LOCK = threading.RLock()
VIDEO_TASK_CONFIG_TTL_SECONDS = 7 * 24 * 3600
VIDEO_TASK_CONFIG_TOKEN_PREFIX = 'cfg1'
VIDEO_TASK_CONFIG_SECRET = os.environ.get(
    'VIDEO_TASK_CONFIG_SECRET', 'magic-mirror-config-secret'
)


def _cleanup_expired_progress() -> None:
    """Remove finished task progress records older than _PROGRESS_TTL_SECONDS."""
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
                elif now - float(finished_at) > _PROGRESS_TTL_SECONDS:
                    to_remove.append(task_id)
        for task_id in to_remove:
            VIDEO_TASK_PROGRESS.pop(task_id, None)
    with VIDEO_TASK_CANCELLED_LOCK:
        for task_id in to_remove:
            VIDEO_TASK_CANCELLED.discard(task_id)


def _maybe_gc_progress() -> None:
    """Throttled GC for progress records."""
    global _LAST_PROGRESS_GC_AT
    now = time.time()
    with _PROGRESS_GC_LOCK:
        if now - _LAST_PROGRESS_GC_AT < _PROGRESS_GC_INTERVAL:
            return
        _LAST_PROGRESS_GC_AT = now
    try:
        _cleanup_expired_progress()
    except Exception:
        pass


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
    _maybe_gc_progress()
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id)
        if not state:
            return {'status': 'idle', 'progress': 0, 'etaSeconds': None, 'stage': None}
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


def _cleanup_cancelled_video_result(result) -> None:
    """Delete a completed output when cancellation wins the finish race."""
    path = str(result or '')
    if not path or not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _run_image_task_once(task_id: str, task_callable):
    """Claim an image task ID and keep cancellation generation-safe."""
    normalized_task_id = str(task_id)
    cancel_event = threading.Event()
    _maybe_gc_progress()
    with VIDEO_TASK_START_LOCK:
        if (
            normalized_task_id in IMAGE_TASKS
            or VIDEO_TASK_EXECUTOR.is_active(normalized_task_id)
            or normalized_task_id in VIDEO_TASK_PROGRESS
        ):
            return None, RuntimeError('task-already-running')
        IMAGE_TASKS[normalized_task_id] = cancel_event

    def _guarded_task():
        if cancel_event.is_set():
            raise RuntimeError('cancelled')
        return task_callable()

    try:
        return AsyncTask.run(_guarded_task, task_id=normalized_task_id)
    finally:
        with VIDEO_TASK_START_LOCK:
            if IMAGE_TASKS.get(normalized_task_id) is cancel_event:
                IMAGE_TASKS.pop(normalized_task_id, None)


def _run_video_task_async(task_id: str, task_callable, on_completion):
    """Queue a video task in the bounded background executor."""

    def _completion(res, err):
        try:
            on_completion(res, err)
        except Exception:
            print(
                '[ERROR] video task completion callback failed:\n',
                traceback.format_exc(),
            )
        finally:
            _clear_video_task_cancelled(task_id)

    VIDEO_TASK_EXECUTOR.submit(
        task_id,
        task_callable,
        _completion,
        on_cancel_result=_cleanup_cancelled_video_result,
    )


def _clone_json_payload(payload):
    """Deep clone a JSON-serializable payload."""
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _build_video_task_config_token(payload: dict) -> str:
    """Build a signed config token for a video task."""
    return build_video_task_config_token(payload, VIDEO_TASK_CONFIG_SECRET)


def _parse_video_task_config_token(config_id: str):
    """Parse and verify a video task config token."""
    return parse_video_task_config_token(
        str(config_id),
        VIDEO_TASK_CONFIG_SECRET,
        legacy_ttl_seconds=VIDEO_TASK_CONFIG_TTL_SECONDS,
    )


def _cleanup_video_task_configs():
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


def _store_video_task_config(payload: dict, config_id: str | None = None) -> str:
    """Store a video task configuration."""
    _cleanup_video_task_configs()
    if not isinstance(payload, dict):
        raise RuntimeError('missing-params')

    next_id = str(config_id or _build_video_task_config_token(payload))
    with VIDEO_TASK_CONFIGS_LOCK:
        VIDEO_TASK_CONFIGS[next_id] = {
            'createdAt': time.time(),
            'config': _clone_json_payload(payload),
        }
    return next_id


def _get_video_task_config(config_id: str):
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


def _build_video_task_config_payload(
    *,
    input_video: str,
    target_face: str | None = None,
    target_faces: list[dict] | None = None,
    deep_swap_mode: bool = False,
    segment_duration_sec: int = 12,
    segment_overlap_frames: int = 6,
    face_sources: list[dict] | None = None,
    regions=None,
    key_frame_ms: int = 0,
    use_gpu: bool = False,
    gpu_provider: str = 'auto',
):
    """Build the config payload for a video task."""
    payload = {
        'inputVideo': {
            'path': input_video,
            'sha256': compute_file_sha256(input_video),
        },
        'regions': regions if isinstance(regions, list) else None,
        'keyFrameMs': max(0, int(key_frame_ms or 0)),
        'useGpu': bool(use_gpu),
        'gpuProvider': str(gpu_provider or 'auto').strip().lower(),
    }

    if isinstance(face_sources, list) and len(face_sources) > 0:
        normalized_sources = []
        for source in face_sources:
            if not isinstance(source, dict):
                raise RuntimeError('missing-face-sources')
            source_id = source.get('id')
            source_path = source.get('path')
            if source_id is None or not source_path:
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
        if not target_face:
            raise RuntimeError('missing-params')
        payload['targetFace'] = {
            'path': target_face,
            'sha256': compute_file_sha256(target_face),
        }

    return payload


def _ensure_video_task_config_matches(
    config: dict,
    input_video: str,
    *,
    target_face: str | None = None,
    target_face_map: dict[str, str] | None = None,
    source_map: dict[str, str] | None = None,
):
    """Ensure a video task config matches expectations."""
    expected_input_sha256 = get_expected_input_video_sha256(config)
    if expected_input_sha256 and not verify_file_sha256(
        input_video, expected_input_sha256
    ):
        raise RuntimeError('config-mismatch')

    expected_target_sha256 = get_expected_target_face_sha256(config)
    if expected_target_sha256:
        if not target_face or not verify_file_sha256(
            target_face, expected_target_sha256
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


def _ext(path: str) -> str:
    """Get the file extension from a path."""
    return os.path.splitext(path)[1].lower()


def _simplify_task_error(err: object) -> str:
    """把内部异常/堆栈信息收敛成前端可用的错误码，避免泄漏本地路径等细节。"""
    msg = (str(err) if err is not None else '').lower()
    codes = [
        'missing-params',
        'missing-face-sources',
        'invalid-face-source-binding',
        'face-source-not-found',
        'file-not-found',
        'unsupported-image-format',
        'unsupported-video-format',
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


def _validate_file(path: str, allowed_exts: set[str], *, missing_code: str):
    """Validate a file exists and has an allowed extension."""
    if not path:
        raise RuntimeError('missing-params')
    if not os.path.exists(path):
        raise FileNotFoundError('file-not-found')
    if _ext(path) not in allowed_exts:
        raise RuntimeError(missing_code)






@app.get('/status')
def status():
    """Return the server status."""
    return {'status': 'running'}


@app.post('/prepare')
def prepare(response: Response):
    """Prepare a file for face detection."""
    return {'success': load_models()}


@app.post('/task')
def create_task(response: Response, body: dict = Body(default_factory=dict)):
    """Create a new face swap task."""
    try:
        raw_task_id = body.get('id')
        task_id = str(raw_task_id).strip() if raw_task_id is not None else ''
        input_image = body.get('inputImage')
        target_face = body.get('targetFace')
        regions = body.get('regions')
        face_sources = body.get('faceSources')
        target_faces = body.get('targetFaces')
        deep_swap_mode = bool(body.get('deepSwapMode', False))
        has_face_sources = 'faceSources' in body
        has_target_faces = 'targetFaces' in body or deep_swap_mode

        if not all([task_id, input_image]):
            response.status_code = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_image,
                ALLOWED_IMAGE_EXTS,
                missing_code='unsupported-image-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

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
                source_path = source.get('path')
                if not source_id or not source_path:
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status_code = 400
                    return {'error': _simplify_task_error(e)}
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

                res, err = _run_image_task_once(
                    task_id,
                    lambda: swap_face_regions_by_sources(
                        input_image, source_map, regions
                    ),
                )
            else:
                fallback_face = next(iter(source_map.values()))
                res, err = _run_image_task_once(
                    task_id,
                    lambda: swap_face(input_image, fallback_face),
                )
        elif has_target_faces:
            if not isinstance(target_faces, list) or len(target_faces) == 0:
                response.status_code = 400
                return {'error': 'missing-params'}

            target_face_paths = []
            for target in target_faces:
                if isinstance(target, str):
                    target_path = target
                elif isinstance(target, dict):
                    target_path = target.get('path')
                else:
                    target_path = None
                if not target_path:
                    response.status_code = 400
                    return {'error': 'missing-params'}
                try:
                    _validate_file(
                        target_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status_code = 400
                    return {'error': _simplify_task_error(e)}
                target_face_paths.append(target_path)

            res, err = _run_image_task_once(
                task_id,
                lambda: swap_face_deep(input_image, target_face_paths, regions=regions),
            )
        else:
            if not target_face:
                response.status_code = 400
                return {'error': 'missing-params'}

            try:
                _validate_file(
                    target_face,
                    ALLOWED_IMAGE_EXTS,
                    missing_code='unsupported-image-format',
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

            if regions:
                res, err = _run_image_task_once(
                    task_id,
                    lambda: swap_face_regions(input_image, target_face, regions),
                )
            else:
                res, err = _run_image_task_once(
                    task_id,
                    lambda: swap_face(input_image, target_face),
                )

        if res:
            return {'result': res}

        error_code = _simplify_task_error(err)
        response.status_code = 409 if error_code == 'task-already-running' else 500
        return {'error': error_code}

    except Exception as e:
        print('[ERROR] create_task failed:', str(e), '\n', traceback.format_exc())
        response.status_code = 500
        return {'error': _simplify_task_error(e)}


@app.post('/task/detect-faces')
def detect_faces_for_image(
    response: Response,
    body: dict = Body(default_factory=dict),
):
    """Detect faces in an uploaded image."""
    try:
        input_image = body.get('inputImage')
        regions = body.get('regions')

        if not input_image:
            response.status_code = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_image,
                ALLOWED_IMAGE_EXTS,
                missing_code='unsupported-image-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status_code = 400
            return {'error': 'missing-params'}

        result = detect_face_boxes_in_image(input_image, regions=regions)
        return {'regions': result}
    except Exception as e:
        response.status_code = 500
        return {'error': _simplify_task_error(e)}


@app.post('/task/video/detect-faces')
def detect_faces_for_video(
    response: Response,
    body: dict = Body(default_factory=dict),
):
    """Detect faces in a video file."""
    try:
        input_video = body.get('inputVideo')
        key_frame_ms = body.get('keyFrameMs', 0)
        regions = body.get('regions')

        if not input_video:
            response.status_code = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_video,
                ALLOWED_VIDEO_EXTS,
                missing_code='unsupported-video-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status_code = 400
            return {'error': 'missing-params'}

        try:
            key_frame_ms = int(float(key_frame_ms or 0))
        except (TypeError, ValueError):
            key_frame_ms = 0

        result = detect_face_boxes_in_video(
            input_video,
            key_frame_ms=max(0, key_frame_ms),
            regions=regions,
        )
        return result
    except Exception as e:
        response.status_code = 500
        return {'error': _simplify_task_error(e)}


@app.get('/task/video/gpu-modes')
def get_video_gpu_modes(response: Response):
    """Return available GPU acceleration modes."""
    try:
        return get_gpu_acceleration_modes()
    except Exception as e:
        response.status_code = 500
        return {
            'modes': [{'id': 'cpu', 'name': 'CPU'}],
            'availableProviders': [],
            'error': _simplify_task_error(e),
        }


@app.post('/task/video')
def create_video_task(response: Response, body: dict = Body(default_factory=dict)):
    """Create a new video face swap task."""
    task_id = None
    try:
        raw_task_id = body.get('id')
        task_id = str(raw_task_id).strip() if raw_task_id is not None else ''
        input_video = body.get('inputVideo')
        target_face = body.get('targetFace')
        target_faces = body.get('targetFaces')
        regions = body.get('regions')
        face_sources = body.get('faceSources')
        deep_swap_mode = bool(body.get('deepSwapMode', False))
        segment_duration_sec = body.get('segmentDurationSec', 12)
        segment_overlap_frames = body.get('segmentOverlapFrames', 6)
        config_id = body.get('configId')
        generate_config_id = bool(body.get('generateConfigId', False))
        dry_run_config_only = bool(body.get('dryRunConfigOnly', False))
        key_frame_ms = body.get('keyFrameMs', 0)
        use_gpu = body.get('useGpu', False)
        gpu_provider = str(body.get('gpuProvider', 'auto') or 'auto').strip().lower()

        stored_config = None
        if config_id:
            stored_config = _get_video_task_config(str(config_id))
            if not isinstance(stored_config, dict):
                response.status_code = 400
                return {'error': 'config-not-found'}

            if target_face is None:
                target_face = _extract_stored_path(stored_config.get('targetFace'))
            if target_faces is None and 'targetFaces' in stored_config:
                target_faces = _extract_stored_face_sources(
                    stored_config.get('targetFaces')
                )
            if face_sources is None and 'faceSources' in stored_config:
                face_sources = _extract_stored_face_sources(
                    stored_config.get('faceSources')
                )
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

        if not all([task_id, input_video]):
            response.status_code = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_video,
                ALLOWED_VIDEO_EXTS,
                missing_code='unsupported-video-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        source_map = None
        target_face_path = None
        target_face_map = None
        target_face_items = None

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
                source_path = source.get('path')
                if not source_id or not source_path:
                    response.status_code = 400
                    return {'error': 'missing-face-sources'}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status_code = 400
                    return {'error': _simplify_task_error(e)}
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
            if not isinstance(target_faces, list) or len(target_faces) == 0:
                response.status_code = 400
                return {'error': 'missing-params'}

            target_face_items = []
            target_face_map = {}
            for index, target in enumerate(target_faces):
                if isinstance(target, str):
                    target_id = f'target-{index + 1}'
                    target_path = target
                elif isinstance(target, dict):
                    target_id = str(target.get('id') or f'target-{index + 1}')
                    target_path = target.get('path')
                else:
                    target_id = None
                    target_path = None

                if not target_id or not target_path:
                    response.status_code = 400
                    return {'error': 'missing-params'}

                try:
                    _validate_file(
                        target_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status_code = 400
                    return {'error': _simplify_task_error(e)}

                target_face_items.append({'id': str(target_id), 'path': target_path})
                target_face_map[str(target_id)] = target_path
        else:
            if not target_face:
                response.status_code = 400
                return {'error': 'missing-params'}

            target_face_path = target_face
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
                    input_video,
                    target_face=target_face_path,
                    target_face_map=target_face_map,
                    source_map=source_map,
                )
            except RuntimeError as e:
                response.status_code = 400
                return {'error': _simplify_task_error(e)}

        try:
            config_payload = _build_video_task_config_payload(
                input_video=input_video,
                target_face=target_face_path,
                target_faces=target_face_items if has_target_faces else None,
                deep_swap_mode=deep_swap_mode,
                segment_duration_sec=segment_duration_sec,
                segment_overlap_frames=segment_overlap_frames,
                face_sources=face_sources if has_face_sources else None,
                regions=regions,
                key_frame_ms=key_frame_ms,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status_code = 400
            return {'error': _simplify_task_error(e)}

        active_config_id = None
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

        # 先设置初始状态
        # 定义回调函数（必须在 task_callable 之前定义）
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
                    # 计算剩余时间：(总帧数 - 已处理帧数) / 处理速度
                    frames_remaining = total_frames - frame_count
                    processing_speed = frame_count / elapsed_seconds  # 帧/秒
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

        # 现在定义 task_callable，此时回调函数已经存在
        if has_face_sources:
            task_callable = lambda cancel_event: swap_face_video_by_sources(
                input_video,
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
                input_video,
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
                input_video,
                target_face_path,
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                cancel_event=cancel_event,
            )
        # 使用异步任务处理，避免阻塞请求线程
        # 前端通过轮询获取进度和最终结果

        def _on_completion(res, err):
            """Handle completion events after video processing."""
            if _is_video_task_cancelled(task_id):
                return

            if res:
                _set_video_task_progress(
                    task_id,
                    status='success',
                    progress=100,
                    etaSeconds=0,
                    stage='done',
                    error=None,
                    result=res,
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

        _maybe_gc_progress()
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
                    result=None,
                )
                _clear_video_task_cancelled(task_id)
                _run_video_task_async(task_id, task_callable, _on_completion)
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


@app.get('/task/video/progress/{task_id}')
def get_video_task_progress(task_id: str, response: Response):
    """Return the progress of a video task."""
    response.headers['Cache-Control'] = 'no-store'
    return _get_video_task_progress(task_id)


@app.delete('/task/{task_id}')
def cancel_task(task_id: str):
    """Cancel a running task."""
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

@app.websocket('/task/video/ws/{task_id}')
async def video_task_ws(websocket: WebSocket, task_id: str):
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
