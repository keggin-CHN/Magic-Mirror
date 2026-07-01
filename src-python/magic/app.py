import json
import os
import threading
import time
import traceback

from async_tasks import AsyncTask
from bottle import Bottle, request, response

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

app = Bottle()

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


def _run_video_task_async(task_id: str, task_callable, on_completion):
    """Run a video task asynchronously in a background thread."""

    def _worker():
        """Background worker thread for async tasks."""
        res = None
        err = None
        try:
            res = task_callable()
        except Exception as e:
            err = e
        try:
            on_completion(res, err)
        except Exception:
            print(
                '[ERROR] video task completion callback failed:\n',
                traceback.format_exc(),
            )
        finally:
            _clear_video_task_cancelled(task_id)

    thread = threading.Thread(target=_worker, name=f'VideoTask-{task_id}', daemon=True)
    thread.start()


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


# https://github.com/bottlepy/bottle/issues/881#issuecomment-244024649
app.plugins[0].json_dumps = lambda *args, **kwargs: json.dumps(
    *args, ensure_ascii=False, **kwargs
).encode('utf8')


# Enable CORS
@app.hook('after_request')
def enable_cors():
    """Configure CORS headers for all responses."""
    response.set_header('Access-Control-Allow-Origin', '*')
    response.set_header('Access-Control-Allow-Methods', '*')
    response.set_header('Access-Control-Allow-Headers', '*')


@app.route('<path:path>', method=['OPTIONS'])
def handle_options(path):
    """Handle CORS preflight OPTIONS requests."""
    response.status = 200
    return 'MagicMirror ✨'


@app.get('/status')
def status():
    """Return the server status."""
    return {'status': 'running'}


@app.route('/prepare', method=['POST', 'OPTIONS'])
def prepare():
    """Prepare a file for face detection."""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        return {}

    return {'success': load_models()}


@app.route('/task', method=['POST', 'OPTIONS'])
def create_task():
    """Create a new face swap task."""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        return {}

    try:
        body = request.json or {}
        task_id = body.get('id')
        input_image = body.get('inputImage')
        target_face = body.get('targetFace')
        regions = body.get('regions')
        face_sources = body.get('faceSources')
        target_faces = body.get('targetFaces')
        deep_swap_mode = bool(body.get('deepSwapMode', False))
        has_face_sources = 'faceSources' in body
        has_target_faces = 'targetFaces' in body or deep_swap_mode

        if not all([task_id, input_image]):
            response.status = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_image,
                ALLOWED_IMAGE_EXTS,
                missing_code='unsupported-image-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {'error': _simplify_task_error(e)}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status = 400
                return {'error': 'missing-face-sources'}

            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status = 400
                    return {'error': 'missing-face-sources'}
                source_id = source.get('id')
                source_path = source.get('path')
                if not source_id or not source_path:
                    response.status = 400
                    return {'error': 'missing-face-sources'}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {'error': _simplify_task_error(e)}
                source_map[str(source_id)] = source_path

            if regions:
                if not isinstance(regions, list):
                    response.status = 400
                    return {'error': 'invalid-face-source-binding'}

                for region in regions:
                    if not isinstance(region, dict):
                        response.status = 400
                        return {'error': 'invalid-face-source-binding'}
                    source_id = region.get('faceSourceId')
                    if not source_id or str(source_id) not in source_map:
                        response.status = 400
                        return {'error': 'invalid-face-source-binding'}

                res, err = AsyncTask.run(
                    lambda: swap_face_regions_by_sources(
                        input_image, source_map, regions
                    ),
                    task_id=task_id,
                )
            else:
                fallback_face = next(iter(source_map.values()))
                res, err = AsyncTask.run(
                    lambda: swap_face(input_image, fallback_face),
                    task_id=task_id,
                )
        elif has_target_faces:
            if not isinstance(target_faces, list) or len(target_faces) == 0:
                response.status = 400
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
                    response.status = 400
                    return {'error': 'missing-params'}
                try:
                    _validate_file(
                        target_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {'error': _simplify_task_error(e)}
                target_face_paths.append(target_path)

            res, err = AsyncTask.run(
                lambda: swap_face_deep(input_image, target_face_paths, regions=regions),
                task_id=task_id,
            )
        else:
            if not target_face:
                response.status = 400
                return {'error': 'missing-params'}

            try:
                _validate_file(
                    target_face,
                    ALLOWED_IMAGE_EXTS,
                    missing_code='unsupported-image-format',
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {'error': _simplify_task_error(e)}

            if regions:
                res, err = AsyncTask.run(
                    lambda: swap_face_regions(input_image, target_face, regions),
                    task_id=task_id,
                )
            else:
                res, err = AsyncTask.run(
                    lambda: swap_face(input_image, target_face),
                    task_id=task_id,
                )

        if res:
            return {'result': res}

        response.status = 500
        return {'error': _simplify_task_error(err)}

    except Exception as e:
        print('[ERROR] create_task failed:', str(e), '\n', traceback.format_exc())
        response.status = 500
        return {'error': _simplify_task_error(e)}


@app.route('/task/detect-faces', method=['POST', 'OPTIONS'])
def detect_faces_for_image():
    """Detect faces in an uploaded image."""
    if request.method == 'OPTIONS':
        return {}

    try:
        body = request.json or {}
        input_image = body.get('inputImage')
        regions = body.get('regions')

        if not input_image:
            response.status = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_image,
                ALLOWED_IMAGE_EXTS,
                missing_code='unsupported-image-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {'error': _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status = 400
            return {'error': 'missing-params'}

        result = detect_face_boxes_in_image(input_image, regions=regions)
        return {'regions': result}
    except Exception as e:
        response.status = 500
        return {'error': _simplify_task_error(e)}


@app.route('/task/video/detect-faces', method=['POST', 'OPTIONS'])
def detect_faces_for_video():
    """Detect faces in a video file."""
    if request.method == 'OPTIONS':
        return {}

    try:
        body = request.json or {}
        input_video = body.get('inputVideo')
        key_frame_ms = body.get('keyFrameMs', 0)
        regions = body.get('regions')

        if not input_video:
            response.status = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_video,
                ALLOWED_VIDEO_EXTS,
                missing_code='unsupported-video-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {'error': _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status = 400
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
        response.status = 500
        return {'error': _simplify_task_error(e)}


@app.route('/task/video/gpu-modes', method=['GET', 'OPTIONS'])
def get_video_gpu_modes():
    """Return available GPU acceleration modes."""
    if request.method == 'OPTIONS':
        return {}

    try:
        return get_gpu_acceleration_modes()
    except Exception as e:
        response.status = 500
        return {
            'modes': [{'id': 'cpu', 'name': 'CPU'}],
            'availableProviders': [],
            'error': _simplify_task_error(e),
        }


@app.route('/task/video', method=['POST', 'OPTIONS'])
def create_video_task():
    """Create a new video face swap task."""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        return {}

    task_id = None
    try:
        body = request.json or {}
        task_id = body.get('id')
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
                response.status = 400
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
            response.status = 400
            return {'error': 'missing-params'}

        try:
            _validate_file(
                input_video,
                ALLOWED_VIDEO_EXTS,
                missing_code='unsupported-video-format',
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {'error': _simplify_task_error(e)}

        source_map = None
        target_face_path = None
        target_face_map = None
        target_face_items = None

        if has_face_sources and has_target_faces:
            response.status = 400
            return {'error': 'missing-params'}

        if has_face_sources:
            if not isinstance(face_sources, list) or len(face_sources) == 0:
                response.status = 400
                return {'error': 'missing-face-sources'}

            source_map = {}
            for source in face_sources:
                if not isinstance(source, dict):
                    response.status = 400
                    return {'error': 'missing-face-sources'}
                source_id = source.get('id')
                source_path = source.get('path')
                if not source_id or not source_path:
                    response.status = 400
                    return {'error': 'missing-face-sources'}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {'error': _simplify_task_error(e)}
                source_map[str(source_id)] = source_path

            if not isinstance(regions, list) or len(regions) == 0:
                response.status = 400
                return {'error': 'invalid-face-source-binding'}

            for region in regions:
                if not isinstance(region, dict):
                    response.status = 400
                    return {'error': 'invalid-face-source-binding'}
                source_id = region.get('faceSourceId')
                if not source_id or str(source_id) not in source_map:
                    response.status = 400
                    return {'error': 'invalid-face-source-binding'}
        elif has_target_faces:
            if not isinstance(target_faces, list) or len(target_faces) == 0:
                response.status = 400
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
                    response.status = 400
                    return {'error': 'missing-params'}

                try:
                    _validate_file(
                        target_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code='unsupported-image-format',
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {'error': _simplify_task_error(e)}

                target_face_items.append({'id': str(target_id), 'path': target_path})
                target_face_map[str(target_id)] = target_path
        else:
            if not target_face:
                response.status = 400
                return {'error': 'missing-params'}

            target_face_path = target_face
            try:
                _validate_file(
                    target_face_path,
                    ALLOWED_IMAGE_EXTS,
                    missing_code='unsupported-image-format',
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
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
                response.status = 400
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
            response.status = 400
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
            task_callable = lambda: swap_face_video_by_sources(
                input_video,
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
            )
        else:
            task_callable = lambda: swap_face_video(
                input_video,
                target_face_path,
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
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

        try:
            _run_video_task_async(task_id, task_callable, _on_completion)
        except Exception as e:
            _set_video_task_progress(
                task_id,
                status='failed',
                stage='failed',
                error=_simplify_task_error(e),
                etaSeconds=None,
            )
            response.status = 500
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
        response.status = 500
        return {'error': _simplify_task_error(e)}


@app.get('/task/video/progress/<task_id>')
def get_video_task_progress(task_id):
    """Return the progress of a video task."""
    response.set_header('Cache-Control', 'no-store')
    return _get_video_task_progress(task_id)


@app.delete('/task/<task_id>')
def cancel_task(task_id):
    """Cancel a running task."""
    AsyncTask.cancel(task_id)
    _mark_video_task_cancelled(task_id)
    _set_video_task_progress(
        task_id,
        status='cancelled',
        stage='cancelled',
        etaSeconds=None,
        error='cancelled',
    )
    return {'success': True}
