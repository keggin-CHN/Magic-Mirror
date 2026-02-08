import json
import os
import threading
import traceback

from async_tasks import AsyncTask
from bottle import Bottle, request, response

from .face import (
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


VIDEO_TASK_PROGRESS = {}
VIDEO_TASK_PROGRESS_LOCK = threading.RLock()


def _set_video_task_progress(task_id: str, **updates):
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id, {})
        state.update(updates)
        VIDEO_TASK_PROGRESS[task_id] = state


def _get_video_task_progress(task_id: str):
    with VIDEO_TASK_PROGRESS_LOCK:
        state = VIDEO_TASK_PROGRESS.get(task_id)
        if not state:
            return {"status": "idle", "progress": 0, "etaSeconds": None, "stage": None}
        return state.copy()


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _simplify_task_error(err: object) -> str:
    """把内部异常/堆栈信息收敛成前端可用的错误码，避免泄漏本地路径等细节。"""
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


def _validate_file(path: str, allowed_exts: set[str], *, missing_code: str):
    if not path:
        raise RuntimeError("missing-params")
    if not os.path.exists(path):
        raise FileNotFoundError("file-not-found")
    if _ext(path) not in allowed_exts:
        raise RuntimeError(missing_code)

# https://github.com/bottlepy/bottle/issues/881#issuecomment-244024649
app.plugins[0].json_dumps = lambda *args, **kwargs: json.dumps(
    *args, ensure_ascii=False, **kwargs
).encode("utf8")


# Enable CORS
@app.hook("after_request")
def enable_cors():
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header("Access-Control-Allow-Methods", "*")
    response.set_header("Access-Control-Allow-Headers", "*")


@app.route("<path:path>", method=["GET", "OPTIONS"])
def handle_options(path):
    response.status = 200
    return "MagicMirror ✨"


@app.get("/status")
def status():
    return {"status": "running"}


@app.route("/prepare", method=["POST", "OPTIONS"])
def prepare():
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        return {}
    
    return {"success": load_models()}


@app.route("/task", method=["POST", "OPTIONS"])
def create_task():
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        return {}

    try:
        body = request.json or {}
        task_id = body.get("id")
        input_image = body.get("inputImage")
        target_face = body.get("targetFace")
        regions = body.get("regions")
        face_sources = body.get("faceSources")
        has_face_sources = "faceSources" in body

        if not all([task_id, input_image]):
            response.status = 400
            return {"error": "missing-params"}

        try:
            _validate_file(
                input_image,
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
                source_path = source.get("path")
                if not source_id or not source_path:
                    response.status = 400
                    return {"error": "missing-face-sources"}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code="unsupported-image-format",
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {"error": _simplify_task_error(e)}
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
        else:
            if not target_face:
                response.status = 400
                return {"error": "missing-params"}

            try:
                _validate_file(
                    target_face,
                    ALLOWED_IMAGE_EXTS,
                    missing_code="unsupported-image-format",
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

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
            return {"result": res}

        response.status = 500
        return {"error": _simplify_task_error(err)}

    except Exception as e:
        print("[ERROR] create_task failed:", str(e), "\n", traceback.format_exc())
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.route("/task/detect-faces", method=["POST", "OPTIONS"])
def detect_faces_for_image():
    if request.method == "OPTIONS":
        return {}

    try:
        body = request.json or {}
        input_image = body.get("inputImage")
        regions = body.get("regions")

        if not input_image:
            response.status = 400
            return {"error": "missing-params"}

        try:
            _validate_file(
                input_image,
                ALLOWED_IMAGE_EXTS,
                missing_code="unsupported-image-format",
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {"error": _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status = 400
            return {"error": "missing-params"}

        result = detect_face_boxes_in_image(input_image, regions=regions)
        return {"regions": result}
    except Exception as e:
        response.status = 500
        return {"error": _simplify_task_error(e)}


@app.route("/task/video/detect-faces", method=["POST", "OPTIONS"])
def detect_faces_for_video():
    if request.method == "OPTIONS":
        return {}

    try:
        body = request.json or {}
        input_video = body.get("inputVideo")
        key_frame_ms = body.get("keyFrameMs", 0)
        regions = body.get("regions")

        if not input_video:
            response.status = 400
            return {"error": "missing-params"}

        try:
            _validate_file(
                input_video,
                ALLOWED_VIDEO_EXTS,
                missing_code="unsupported-video-format",
            )
        except (RuntimeError, FileNotFoundError) as e:
            response.status = 400
            return {"error": _simplify_task_error(e)}

        if regions is not None and not isinstance(regions, list):
            response.status = 400
            return {"error": "missing-params"}

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
        return {"error": _simplify_task_error(e)}


@app.route("/task/video", method=["POST", "OPTIONS"])
def create_video_task():
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        return {}

    task_id = None
    try:
        body = request.json or {}
        task_id = body.get("id")
        input_video = body.get("inputVideo")
        target_face = body.get("targetFace")
        regions = body.get("regions")
        face_sources = body.get("faceSources")
        key_frame_ms = body.get("keyFrameMs", 0)
        use_gpu = body.get("useGpu", False)
        has_face_sources = "faceSources" in body

        print("[API] 收到视频换脸请求:")
        print(f"  - task_id: {task_id}")
        print(f"  - input_video: {input_video}")
        print(f"  - target_face: {target_face}")
        print(f"  - has_face_sources: {has_face_sources}")
        print(f"  - key_frame_ms: {key_frame_ms}")
        print(f"  - use_gpu: {use_gpu}")

        if not all([task_id, input_video]):
            response.status = 400
            return {"error": "missing-params"}

        try:
            _validate_file(
                input_video,
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
                source_path = source.get("path")
                if not source_id or not source_path:
                    response.status = 400
                    return {"error": "missing-face-sources"}
                try:
                    _validate_file(
                        source_path,
                        ALLOWED_IMAGE_EXTS,
                        missing_code="unsupported-image-format",
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    response.status = 400
                    return {"error": _simplify_task_error(e)}
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

        # 先设置初始状态
        _set_video_task_progress(
            task_id,
            status="running",
            progress=0,
            etaSeconds=None,
            frameCount=0,
            totalFrames=0,
            error=None,
            result=None,
        )

        # 定义回调函数（必须在 task_callable 之前定义）
        def _on_stage(stage: str):
            # 简化：不再更新stage，只在日志中记录
            print(f"[INFO] 视频处理阶段: {stage}")

        def _on_progress(frame_count: int, total_frames: int, elapsed_seconds: float):
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
                status="running",
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
            )
        else:
            if not target_face:
                response.status = 400
                return {"error": "missing-params"}

            try:
                _validate_file(
                    target_face,
                    ALLOWED_IMAGE_EXTS,
                    missing_code="unsupported-image-format",
                )
            except (RuntimeError, FileNotFoundError) as e:
                response.status = 400
                return {"error": _simplify_task_error(e)}

            task_callable = lambda: swap_face_video(
                input_video,
                target_face,
                progress_callback=_on_progress,
                stage_callback=_on_stage,
                use_gpu=use_gpu,
            )
# 使用异步任务处理，避免阻塞请求线程
# 前端通过轮询获取进度和最终结果

        
        def _on_completion(res, err):
            if res:
                print(f"[API] 视频换脸任务完成: {res}")
                _set_video_task_progress(
                    task_id,
                    status="success",
                    progress=100,
                    etaSeconds=0,
                    error=None,
                    result=res,
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
            AsyncTask.run_async(task_callable, task_id=task_id, on_completion=_on_completion)
        except Exception as e:
            print(f"[ERROR] 启动异步任务失败: {str(e)}")
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


@app.get("/task/video/progress/<task_id>")
def get_video_task_progress(task_id):
    return _get_video_task_progress(task_id)


@app.delete("/task/<task_id>")
def cancel_task(task_id):
    AsyncTask.cancel(task_id)
    _set_video_task_progress(
        task_id,
        status="cancelled",
        etaSeconds=None,
        error="cancelled",
    )
    return {"success": True}
