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
            error=None,
            result=None,
            stage="queued",
        )

        # 定义回调函数（必须在 task_callable 之前定义）
        def _on_stage(stage: str):
            _set_video_task_progress(
                task_id,
                status="running",
                stage=stage,
                error=None,
            )

        def _on_progress(frame_count: int, total_frames: int, elapsed_seconds: float):
            progress = 0.0
            eta_seconds = None
            if total_frames and total_frames > 0:
                progress = max(0.0, min(100.0, frame_count / total_frames * 100.0))
                if frame_count > 0:
                    eta_seconds = max(
                        0, int((elapsed_seconds / frame_count) * (total_frames - frame_count))
                    )
            _set_video_task_progress(
                task_id,
                status="running",
                progress=round(progress, 2),
                etaSeconds=eta_seconds,
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

        # Use run_async instead of run to avoid blocking the request thread
        # This allows the request to return immediately with a task ID
        # But wait, the frontend expects the result in the response of createVideoTask?
        # Let's check frontend code.
        # src/hooks/useSwapFace.ts:
        # const { result, error } = await Server.createVideoTask(...)
        # if (result) { setVideoProgress(100); ... }
        
        # So the frontend DOES expect the result in the response.
        # This means the backend MUST block until the task is done.
        
        # If the backend blocks, why is progress not updating?
        # Maybe the GIL?
        # Or maybe the browser limits the number of concurrent connections to the same domain?
        # If createVideoTask takes a long time, the browser might stall other requests (like progress polling)
        # to the same domain (localhost:8023).
        
        # Browsers typically limit to 6 connections per domain.
        # Here we have 1 long-running connection (createVideoTask) and 1 polling connection.
        # It should be fine.
        
        # However, if the server is single-threaded or has a global lock, it might be an issue.
        # server.py uses ThreadingMixIn, so it's multi-threaded.
        # VIDEO_TASK_PROGRESS_LOCK is a RLock.
        
        # Let's look at async_tasks again.
        # AsyncTask.run joins the thread.
        # This blocks the current thread (the request handler thread).
        # This is fine as long as there are other threads available to handle other requests.
        
        # Is it possible that the task itself holds the GIL for too long?
        # swap_face_video calls OpenCV and TinyFace (ONNX Runtime).
        # These should release GIL.
        
        # Wait, I see `app.plugins[0].json_dumps` in app.py.
        # Bottle's default server is single-threaded?
        # server.py uses wsgiref.simple_server.make_server with _ThreadingWSGIServer.
        # So it is multi-threaded.
        
        # Let's try to force the task to run in a separate thread WITHOUT joining, 
        # but we need to return the result.
        # We can't return the result if we don't wait.
        
        # If the issue is that the browser is waiting for the response and somehow blocking the polling?
        # No, that shouldn't happen with fetch.
        
        # Maybe the issue is in the frontend polling loop?
        # const pollProgress = async () => {
        #   while (polling) {
        #     const state = await Server.getVideoTaskProgress(taskId);
        #     ...
        #     await new Promise((resolve) => setTimeout(resolve, 400));
        #   }
        # };
        # const pollPromise = pollProgress();
        # const { result, error } = await Server.createVideoTask(...)
        # polling = false;
        # await pollPromise;
        
        # This looks correct.
        
        # What if we change the backend to return immediately with "queued" status,
        # and let the frontend poll for the result?
        # The frontend code seems to handle "success" status from polling:
        # if (state.status === "running" || state.status === "success") { ... }
        
        # But useSwapFace.ts lines 119-122:
        # if (result) {
        #   setVideoProgress(100);
        #   setVideoEtaSeconds(0);
        #   setVideoStage("done");
        # }
        
        # It relies on the result from createVideoTask to mark as done.
        
        # If we change createVideoTask to be non-blocking:
        # 1. Backend: createVideoTask starts the task and returns { task_id: ... } immediately.
        # 2. Frontend: createVideoTask returns immediately.
        # 3. Frontend: polling loop continues until status is "success" or "failed".
        # 4. Frontend: when "success", it needs to get the result (output path).
        #    The current progress API returns result?
        #    Let's check _get_video_task_progress in app.py.
        #    It returns state.copy().
        #    _set_video_task_progress updates the state.
        #    When task finishes (line 485), it sets result=res.
        #    So yes, the progress API returns the result!
        
        # So we CAN make createVideoTask non-blocking!
        # This would solve any potential blocking issues (browser connection limits, server thread pool exhaustion, etc).
        # And it provides a better UX (immediate feedback that task is accepted).
        
        # Let's modify app.py to use run_async and return immediately.
        
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
                    stage="done",
                )
            else:
                final_error = _simplify_task_error(err)
                _set_video_task_progress(
                    task_id,
                    status="failed",
                    error=final_error,
                    etaSeconds=None,
                    stage="failed",
                )

        AsyncTask.run_async(task_callable, task_id=task_id, on_completion=_on_completion)
        return {"task_id": task_id, "status": "queued"}

    except Exception as e:
        print("[ERROR] create_video_task failed:", str(e), "\n", traceback.format_exc())
        if task_id:
            _set_video_task_progress(
                task_id,
                status="failed",
                error=_simplify_task_error(e),
                etaSeconds=None,
                stage="failed",
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
        stage="cancelled",
    )
    return {"success": True}
