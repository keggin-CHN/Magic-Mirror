import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import cv2
import numpy as np
from tinyface import TinyFace

# 全局 TinyFace 实例（CPU 版本）
_tf = TinyFace()
_tf_lock = threading.RLock()

# GPU 加速的 TinyFace 实例池缓存（按 Provider 复用）
# 结构: {
#   provider: {
#     "instances": [TinyFace, ...],
#     "locks": [RLock, ...],
#   }
# }
_tf_gpu_instances = {}
_tf_gpu_lock = threading.RLock()


_VERBOSE_LOGS = os.environ.get("MAGIC_VERBOSE_LOGS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug_log(message: str):
    """Log a debug message if debug mode is enabled."""
    if _VERBOSE_LOGS:
        print(message)


def _log_error(context: str, error: Exception):
    """记录详细的错误信息"""
    error_msg = f"[ERROR] {context}\n"
    error_msg += f"错误类型: {type(error).__name__}\n"
    error_msg += f"错误信息: {str(error)}\n"
    error_msg += f"堆栈跟踪:\n{traceback.format_exc()}"
    print(error_msg)
    return error_msg


def _clear_queue(q):
    """安全地清空队列"""
    try:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
    except Exception as e:
        print(f"[WARN] 清空队列失败: {str(e)}")


def load_models():
    """Load the face detection and recognition models."""
    try:
        _tf.config.face_detector_model = _get_model_path("scrfd_2.5g.onnx")
        _tf.config.face_embedder_model = _get_model_path("arcface_w600k_r50.onnx")
        _tf.config.face_swapper_model = _get_model_path("inswapper_128_fp16.onnx")
        _tf.config.face_enhancer_model = _get_model_path("gfpgan_1.4.onnx")
        _tf.prepare()
        return True
    except BaseException as _:
        return False


@lru_cache(maxsize=1)
def _get_available_execution_providers():
    """Get available ONNX runtime execution providers."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return list(providers) if providers else []
    except Exception as e:
        print(f"[WARN] 获取 ExecutionProvider 失败: {str(e)}")
        return []


def get_gpu_acceleration_modes():
    """返回当前环境可用的加速模式，供前端在视频换脸前选择。"""
    available_providers = _get_available_execution_providers()
    modes = [{"id": "cpu", "name": "CPU"}]

    if "DmlExecutionProvider" in available_providers:
        modes.append({"id": "directml", "name": "DirectML"})

    if "CUDAExecutionProvider" in available_providers:
        modes.append({"id": "cuda", "name": "CUDA"})

    return {"modes": modes, "availableProviders": available_providers}


def _normalize_gpu_provider(gpu_provider: str):
    """Normalize GPU provider name."""
    mode = (gpu_provider or "auto").strip().lower()
    if mode in {"dml", "directml"}:
        return "directml"
    if mode == "cuda":
        return "cuda"
    if mode == "cpu":
        return "cpu"
    return "auto"


def _resolve_execution_provider(gpu_provider: str):
    """Resolve the execution provider for ONNX runtime."""
    mode = _normalize_gpu_provider(gpu_provider)
    if mode == "cpu":
        return None

    available_providers = _get_available_execution_providers()
    if mode == "cuda":
        candidates = ["CUDAExecutionProvider"]
    elif mode == "directml":
        candidates = ["DmlExecutionProvider"]
    else:
        candidates = ["DmlExecutionProvider", "CUDAExecutionProvider"]

    for provider in candidates:
        if provider in available_providers:
            return provider
    return None

def _resolve_gpu_pool_size(num_workers: int) -> int:
    """根据并发线程数和环境变量推导 GPU 实例池大小。"""
    env_val = os.environ.get("MAGIC_GPU_POOL_SIZE")
    if env_val:
        try:
            forced = int(env_val)
            if forced > 0:
                return max(1, min(forced, 8))
        except Exception:
            print(f"[WARN] 无效 MAGIC_GPU_POOL_SIZE={env_val}，将使用自动策略")
    # 默认最多 4 个实例，避免显存占用过高
    return max(1, min(int(num_workers or 1), 4))


def _init_gpu_models(gpu_provider: str = "auto", pool_size: int = 1):
    """按需初始化指定 Provider 的 GPU 模型实例池，并缓存。"""
    global _tf_gpu_instances

    selected_provider = _resolve_execution_provider(gpu_provider)
    if selected_provider is None:
        return False, None

    target_pool_size = max(1, int(pool_size or 1))

    with _tf_gpu_lock:
        cache = _tf_gpu_instances.get(selected_provider)

        # 兼容旧结构：provider -> TinyFace
        if cache is not None and not isinstance(cache, dict):
            cache = {"instances": [cache], "locks": [threading.RLock()]}
            _tf_gpu_instances[selected_provider] = cache

        if cache is None:
            cache = {"instances": [], "locks": []}
            _tf_gpu_instances[selected_provider] = cache

        instances = cache.get("instances")
        locks = cache.get("locks")
        if not isinstance(instances, list) or not isinstance(locks, list):
            cache["instances"] = []
            cache["locks"] = []
            instances = cache["instances"]
            locks = cache["locks"]

        if len(instances) >= target_pool_size:
            return True, selected_provider

        try:
            for idx in range(len(instances), target_pool_size):
                print(
                    f"[INFO] 正在初始化 GPU 加速模型: {selected_provider} ({idx + 1}/{target_pool_size})"
                )
                tf_gpu = TinyFace()
                tf_gpu.config.face_detector_model = _get_model_path("scrfd_2.5g.onnx")
                tf_gpu.config.face_embedder_model = _get_model_path("arcface_w600k_r50.onnx")
                tf_gpu.config.face_swapper_model = _get_model_path("inswapper_128_fp16.onnx")
                tf_gpu.config.face_enhancer_model = _get_model_path("gfpgan_1.4.onnx")
                tf_gpu.config.execution_providers = [selected_provider, "CPUExecutionProvider"]
                tf_gpu.prepare()
                instances.append(tf_gpu)
                locks.append(threading.RLock())

            print(
                f"[SUCCESS] GPU 模型初始化成功: {selected_provider}, 实例数={len(instances)}"
            )
            return True, selected_provider

        except Exception as e:
            print(f"[ERROR] GPU 模型初始化失败({selected_provider}): {str(e)}")
            print(traceback.format_exc())
            # 若已有可用实例，降级使用现有池
            if len(instances) > 0 and len(instances) == len(locks):
                print(
                    f"[WARN] 使用已初始化的 GPU 实例池继续运行: {selected_provider}, 实例数={len(instances)}"
                )
                return True, selected_provider
            _tf_gpu_instances.pop(selected_provider, None)
            return False, None


def _get_tf_pool(use_gpu=False, gpu_provider="auto", pool_size=1):
    """获取 TinyFace 实例池（CPU 单实例或 GPU 多实例）。"""
    if use_gpu:
        ok, selected_provider = _init_gpu_models(
            gpu_provider=gpu_provider, pool_size=pool_size
        )
        if ok and selected_provider:
            with _tf_gpu_lock:
                cache = _tf_gpu_instances.get(selected_provider) or {}
                instances = list(cache.get("instances") or [])
                locks = list(cache.get("locks") or [])
            if instances and len(instances) == len(locks):
                return list(zip(instances, locks)), True, selected_provider
        print("[WARN] GPU 不可用，回退到 CPU")
    return [(_tf, _tf_lock)], False, None


def _get_tf_instance(use_gpu=False, gpu_provider="auto"):
    """兼容旧调用：获取一个 TinyFace 实例（CPU 或 GPU）。"""
    tf_pool, using_gpu, selected_provider = _get_tf_pool(
        use_gpu=use_gpu, gpu_provider=gpu_provider, pool_size=1
    )
    tf_instance, tf_lock = tf_pool[0]
    return tf_instance, tf_lock, using_gpu, selected_provider


def _emit_stage(stage_callback, stage: str):
    """Emit a stage event to the callback."""
    if stage_callback is None:
        return
    try:
        stage_callback(stage)
    except Exception as e:
        print(f"[WARN] stage_callback failed: {str(e)}")


def swap_face(input_path, face_path):
    """Swap a single face in an image using the target face."""
    save_path = _get_output_file_path(input_path)
    output_img = _swap_face(input_path, face_path)
    return _write_image(save_path, output_img)


def swap_face_regions(input_path, face_path, regions):
    """Swap faces in specific regions of an image."""
    try:
        _debug_log("[DEBUG] swap_face_regions 被调用")
        _debug_log(f"[DEBUG] input_path: {input_path}")
        _debug_log(f"[DEBUG] face_path: {face_path}")
        _debug_log(f"[DEBUG] regions 类型: {type(regions)}, 值: {regions}")

        save_path = _get_output_file_path(input_path)
        input_img = _read_image(input_path)
        height, width = input_img.shape[:2]
        _debug_log(f"[DEBUG] 图片尺寸: {width}x{height}")

        normalized_regions = _normalize_regions(regions, width, height)
        _debug_log(f"[DEBUG] normalized_regions: {normalized_regions}")

        # 未选择/无有效选区：回退全图换脸
        if not normalized_regions:
            _debug_log("[WARN] 无有效选区，回退全图换脸！")
            output_img = _swap_face(input_path, face_path)
            return _write_image(save_path, output_img)

        destination_face = _get_one_face(face_path)
        if destination_face is None:
            raise RuntimeError("no-face-detected")

        output_img = input_img.copy()
        swapped_count = 0

        for x, y, w, h in normalized_regions:
            crop = input_img[y : y + h, x : x + w]
            with _tf_lock:
                reference_face = _tf.get_one_face(crop)
            if reference_face is None:
                continue

            with _tf_lock:
                output_crop = _tf.swap_face(
                    vision_frame=crop,
                    reference_face=reference_face,
                    destination_face=destination_face,
                )
            if output_crop is None:
                continue

            output_img[y : y + h, x : x + w] = output_crop
            swapped_count += 1

        if swapped_count == 0:
            # 用户明确选择了区域，但该区域可能暂时无人脸：
            # 按产品诉求仍需输出文件（保持原图内容），而不是报错中断。
            return _write_image(save_path, output_img)

        return _write_image(save_path, output_img)

    except Exception as e:
        _log_error("swap_face_regions", e)
        raise


def swap_face_regions_by_sources(input_path, face_sources, regions):
    """Swap faces using different source faces for each region."""
    try:
        save_path = _get_output_file_path(input_path)
        input_img = _read_image(input_path)
        height, width = input_img.shape[:2]

        normalized_regions = _normalize_regions_with_face_source(regions, width, height)
        if not normalized_regions:
            raise RuntimeError("invalid-face-source-binding")

        destination_faces = {}
        for source_id, source_path in face_sources.items():
            destination_face = _get_one_face(source_path)
            if destination_face is None:
                raise RuntimeError("no-face-detected")
            destination_faces[str(source_id)] = destination_face

        output_img = input_img.copy()
        swapped_count = 0

        for region in normalized_regions:
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            source_id = region["faceSourceId"]

            destination_face = destination_faces.get(source_id)
            if destination_face is None:
                raise RuntimeError("face-source-not-found")

            crop = input_img[y : y + h, x : x + w]
            with _tf_lock:
                reference_face = _tf.get_one_face(crop)
            if reference_face is None:
                continue

            with _tf_lock:
                output_crop = _tf.swap_face(
                    vision_frame=crop,
                    reference_face=reference_face,
                    destination_face=destination_face,
                )

            if output_crop is None:
                continue

            output_img[y : y + h, x : x + w] = output_crop
            swapped_count += 1

        if swapped_count == 0:
            return _write_image(save_path, output_img)

        return _write_image(save_path, output_img)

    except Exception as e:
        _log_error("swap_face_regions_by_sources", e)
        raise


def swap_face_video(
    input_path,
    face_path,
    regions=None,
    key_frame_ms=0,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
):
    """Swap faces in a video file.

    Detects faces in the input video and replaces them with the specified face image.
    Supports GPU acceleration, custom regions, and progress callbacks.
    """
    try:
        _emit_stage(stage_callback, "validating-input")
        print(
            f"[INFO] 开始视频换脸: input={input_path}, face={face_path}, use_gpu={use_gpu}, gpu_provider={gpu_provider}"
        )

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError("file-not-found")
        if not os.path.exists(face_path):
            raise FileNotFoundError("file-not-found")

        save_path = _get_output_video_path(input_path)
        print(f"[INFO] 输出路径: {save_path}")

        output_path = _swap_face_video(
            input_path,
            face_path,
            save_path,
            regions=regions,
            key_frame_ms=key_frame_ms,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
        )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")

        # 尝试把原视频音轨复用到输出（原视频有音轨时失败应报错，避免静默无声）
        _emit_stage(stage_callback, "muxing-audio")
        _mux_audio_or_raise(input_path, output_path)

        _emit_stage(stage_callback, "finalizing")
        print(f"[SUCCESS] 视频换脸成功: {output_path}")
        return output_path

    except Exception as e:
        _log_error("swap_face_video", e)
        raise


def _swap_face_video(
    input_path,
    face_path,
    save_path,
    regions=None,
    key_frame_ms=0,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
):
    """
    视频换脸处理（支持 GPU 加速和多线程处理池）
    架构：
    - 读取线程：从视频读取帧 -> read_queue
    - 处理线程池：多个线程并行处理帧 -> write_queue
    - 写入线程：从 write_queue 取帧 -> 写入输出视频
    """
    cap = None
    writer = None

    cpu_count = multiprocessing.cpu_count()
    if use_gpu:
        num_workers = max(2, min(cpu_count, 6))
        queue_size = max(8, num_workers * 3)
    else:
        num_workers = max(1, min(cpu_count - 1, 8))
        queue_size = max(5, num_workers * 2)

    print(f"[INFO] 使用 {num_workers} 个处理线程，队列大小: {queue_size}")

    read_queue = queue.Queue(maxsize=queue_size)
    write_queue = queue.PriorityQueue(maxsize=queue_size)

    stop_event = threading.Event()
    processing_error = threading.Lock()
    error_container = {"error": None}
    workers_done_event = threading.Event()
    workers_done_lock = threading.Lock()
    workers_done_count = {"count": 0}

    def _queue_put_with_stop(
        q_obj,
        item,
        *,
        timeout=1,
        warn_prefix="队列已满，等待中...",
    ) -> bool:
        """Put an item in a queue with stop signal support."""
        wait_count = 0
        while not stop_event.is_set():
            try:
                q_obj.put(item, timeout=timeout)
                return True
            except queue.Full:
                wait_count += 1
                if wait_count % 5 == 0:
                    print(f"[WARN] {warn_prefix} (已等待约 {wait_count} 秒)")
        return False

    def _mark_worker_done():
        """Mark a worker thread as done."""
        with workers_done_lock:
            workers_done_count["count"] += 1
            if workers_done_count["count"] >= num_workers:
                workers_done_event.set()

    try:
        _emit_stage(stage_callback, "opening-video")
        print(f"[INFO] 打开视频文件: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")

        _emit_stage(stage_callback, "reading-video-metadata")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
            print(f"[WARN] 无法获取视频FPS，使用默认值: {fps}")
        else:
            print(f"[INFO] 视频FPS: {fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = _resolve_total_frames(input_path, fps, total_frames)

        print(f"[INFO] 视频尺寸: {width}x{height}, 总帧数: {total_frames}")

        if width <= 0 or height <= 0:
            print("[WARN] 无法获取视频尺寸，尝试读取第一帧")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("video-open-failed")
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(f"[INFO] 从第一帧获取尺寸: {width}x{height}")

        normalized_regions = _normalize_regions(regions, width, height) if regions else []
        if regions and not normalized_regions:
            raise RuntimeError("invalid-regions")
        if normalized_regions:
            print(f"[INFO] 单人视频换脸启用区域限制: {len(normalized_regions)} 个选区")

        print(f"[INFO] 创建输出视频: {save_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("video-write-failed")

        _emit_stage(stage_callback, "extracting-target-face")
        print(f"[INFO] 提取目标人脸: {face_path}")

        gpu_pool_size = _resolve_gpu_pool_size(num_workers) if use_gpu else 1
        if use_gpu:
            _emit_stage(stage_callback, "gpu-initializing")
        tf_pool, using_gpu, selected_provider = _get_tf_pool(
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
            pool_size=gpu_pool_size,
        )
        if using_gpu:
            _emit_stage(stage_callback, "gpu-enabled")
            print(
                f"[INFO] 当前 GPU Provider: {selected_provider}, 实例池={len(tf_pool)}, worker={num_workers}"
            )
        elif use_gpu:
            _emit_stage(stage_callback, "gpu-fallback-cpu")
            print(f"[INFO] 已回退 CPU: worker={num_workers}")
        else:
            _emit_stage(stage_callback, "using-cpu")

        bootstrap_tf, bootstrap_lock = tf_pool[0]
        with bootstrap_lock:
            destination_face = bootstrap_tf.get_one_face(_read_image(face_path))
        if destination_face is None:
            raise RuntimeError("no-face-detected")
        print("[SUCCESS] 成功提取目标人脸")

        stats = {
            "frame_count": 0,
            "processed_count": 0,
            "failed_count": 0,
            "start_time": time.time(),
        }
        stats_lock = threading.Lock()
        progress_log_interval = max(30, total_frames // 20) if total_frames > 0 else 300

        def read_frames():
            """Read frames from the input video."""
            try:
                frame_idx = 0
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not _queue_put_with_stop(
                        read_queue,
                        (frame_idx, frame),
                        timeout=1,
                        warn_prefix="读取队列已满，等待处理线程消费",
                    ):
                        return
                    frame_idx += 1

                for _ in range(num_workers):
                    if not _queue_put_with_stop(
                        read_queue,
                        (None, None),
                        timeout=1,
                        warn_prefix="读取队列已满，等待投递结束信号",
                    ):
                        break
            except Exception as e:
                with processing_error:
                    error_container["error"] = e
                print(f"[ERROR] 读取线程异常: {str(e)}")
                stop_event.set()
                _clear_queue(read_queue)

        def process_frames(worker_id):
            """Process frames for face swapping."""
            worker_tf, worker_lock = tf_pool[worker_id % len(tf_pool)]
            try:
                while not stop_event.is_set():
                    try:
                        frame_idx, frame = read_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    if frame_idx is None:
                        break

                    with stats_lock:
                        stats["frame_count"] += 1
                        current_frame = stats["frame_count"]

                    if progress_callback and current_frame % 5 == 0:
                        try:
                            with stats_lock:
                                progress_callback(
                                    frame_count=current_frame,
                                    total_frames=total_frames,
                                    elapsed_seconds=max(
                                        0.0, time.time() - stats["start_time"]
                                    ),
                                )
                        except Exception as e:
                            print(f"[WARN] progress_callback failed: {str(e)}")

                    if current_frame == 1 or current_frame % progress_log_interval == 0:
                        progress = (
                            (current_frame / total_frames * 100)
                            if total_frames > 0
                            else 0
                        )
                        print(
                            f"[PROGRESS] 处理进度: {current_frame}/{total_frames} ({progress:.1f}%) [Worker-{worker_id}]"
                        )

                    try:
                        if normalized_regions:
                            out, swapped_regions = _swap_face_in_regions_for_frame(
                                frame,
                                normalized_regions,
                                worker_tf,
                                worker_lock,
                                destination_face,
                            )
                            if swapped_regions <= 0:
                                with stats_lock:
                                    stats["failed_count"] += 1
                        else:
                            with worker_lock:
                                reference_face = worker_tf.get_one_face(frame)

                            if reference_face is None:
                                if not _queue_put_with_stop(
                                    write_queue,
                                    (frame_idx, frame),
                                    timeout=1,
                                    warn_prefix=f"写入队列已满，Worker-{worker_id} 等待中",
                                ):
                                    break
                                with stats_lock:
                                    stats["failed_count"] += 1
                                continue

                            with worker_lock:
                                output_frame = worker_tf.swap_face(
                                    vision_frame=frame,
                                    reference_face=reference_face,
                                    destination_face=destination_face,
                                )

                            out = output_frame if output_frame is not None else frame
                            with stats_lock:
                                if output_frame is not None:
                                    stats["processed_count"] += 1
                                else:
                                    stats["failed_count"] += 1

                        out = _normalize_output_frame(out, width, height)
                        if not _queue_put_with_stop(
                            write_queue,
                            (frame_idx, out),
                            timeout=1,
                            warn_prefix=f"写入队列已满，Worker-{worker_id} 等待中",
                        ):
                            break

                    except Exception as e:
                        print(f"[WARN] 第{current_frame}帧处理失败: {str(e)}")
                        if not _queue_put_with_stop(
                            write_queue,
                            (frame_idx, frame),
                            timeout=1,
                            warn_prefix=f"写入队列已满，Worker-{worker_id} 等待中",
                        ):
                            break
                        with stats_lock:
                            stats["failed_count"] += 1

            except Exception as e:
                with processing_error:
                    error_container["error"] = e
                print(f"[ERROR] 处理线程 Worker-{worker_id} 异常: {str(e)}")
                stop_event.set()
            finally:
                _mark_worker_done()

        def write_frames():
            """Write processed frames to the output video."""
            try:
                _emit_stage(stage_callback, "processing-video-frames")
                next_frame_idx = 0
                frame_buffer = {}
                frames_written = 0

                while True:
                    if stop_event.is_set() and write_queue.empty():
                        break

                    try:
                        frame_idx, frame = write_queue.get(timeout=1)
                    except queue.Empty:
                        if workers_done_event.is_set():
                            if frame_buffer:
                                for pending_idx in sorted(frame_buffer.keys()):
                                    writer.write(frame_buffer[pending_idx])
                                    frames_written += 1
                                frame_buffer.clear()
                            break
                        continue

                    frame_buffer[frame_idx] = frame
                    while next_frame_idx in frame_buffer:
                        writer.write(frame_buffer.pop(next_frame_idx))
                        frames_written += 1
                        next_frame_idx += 1

                    if (
                        total_frames > 0
                        and frames_written >= total_frames
                        and workers_done_event.is_set()
                    ):
                        break

            except Exception as e:
                with processing_error:
                    error_container["error"] = e
                print(f"[ERROR] 写入线程异常: {str(e)}")
                stop_event.set()

        read_thread = threading.Thread(target=read_frames, name="VideoReader")
        process_threads = [
            threading.Thread(target=process_frames, args=(i,), name=f"VideoProcessor-{i}")
            for i in range(num_workers)
        ]
        write_thread = threading.Thread(target=write_frames, name="VideoWriter")

        read_thread.start()
        for t in process_threads:
            t.start()
        write_thread.start()

        read_thread.join()
        for t in process_threads:
            t.join()
        write_thread.join()

        with processing_error:
            if error_container["error"] is not None:
                raise error_container["error"]

        print("[INFO] 视频处理完成:")
        print(f"  - 总帧数: {stats['frame_count']}")
        print(f"  - 成功换脸: {stats['processed_count']}")
        print(f"  - 跳过/失败: {stats['failed_count']}")

        if progress_callback:
            try:
                final_count = total_frames if total_frames > 0 else stats["frame_count"]
                progress_callback(
                    frame_count=final_count,
                    total_frames=final_count,
                    elapsed_seconds=max(0.0, time.time() - stats["start_time"]),
                )
            except Exception as e:
                print(f"[WARN] progress_callback(final) failed: {str(e)}")

        return save_path

    except Exception as e:
        stop_event.set()
        _log_error("_swap_face_video", e)
        raise

    finally:
        stop_event.set()
        _clear_queue(read_queue)
        _clear_queue(write_queue)

        if cap is not None:
            cap.release()
            print("[INFO] 释放视频读取器")
        if writer is not None:
            writer.release()
            print("[INFO] 释放视频写入器")


def _swap_face(input_path, face_path):
    """Swap a face in a single image."""
    vision = _read_image(input_path)
    reference_face = _get_one_face(input_path)
    destination_face = _get_one_face(face_path)
    if reference_face is None or destination_face is None:
        raise RuntimeError("no-face-detected")
    with _tf_lock:
        out = _tf.swap_face(
            vision_frame=vision,
            reference_face=reference_face,
            destination_face=destination_face,
        )
    if out is None:
        raise RuntimeError("swap-failed")
    return out


def _get_one_face(face_path: str):
    """Detect and return the first face in an image."""
    face_img = _read_image(face_path)
    with _tf_lock:
        return _tf.get_one_face(face_img)


def _read_image(img_path: str):
    """Read an image file as numpy array."""
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("image-decode-failed")

    # 兼容 16-bit PNG/TIFF 等：统一转换成 uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # PNG 可能带 Alpha 或灰度通道，TinyFace 通常期望 BGR 3 通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _write_image(img_path: str, img):
    """Write an image to a file."""
    if img is None:
        raise RuntimeError("swap-failed")

    suffix = (os.path.splitext(img_path)[-1] or ".png").lower()

    def _try_write(path: str, ext: str) -> bool:
        """Try to write an image to a file."""
        ok, buf = cv2.imencode(ext, img)
        if not ok or buf is None:
            return False
        buf.tofile(path)
        return True

    # 先按原扩展名写，失败则回退 PNG（避免 WebP/TIFF 等编码支持不完整导致无输出文件）
    if _try_write(img_path, suffix):
        return img_path

    fallback_path = os.path.splitext(img_path)[0] + ".png"
    if _try_write(fallback_path, ".png"):
        return fallback_path

    raise RuntimeError("output-write-failed")


def _normalize_regions(regions, width, height):
    """Normalize face regions to pixel coordinates."""
    normalized = []
    _debug_log(f"[DEBUG] _normalize_regions: regions={regions}, 图片尺寸={width}x{height}")
    if not regions:
        _debug_log("[DEBUG] regions 为空或 None")
        return normalized
    for i, region in enumerate(regions):
        _debug_log(f"[DEBUG] 处理 region[{i}]: type={type(region)}, value={region}")
        if not isinstance(region, dict):
            _debug_log(f"[DEBUG] region[{i}] 不是 dict，跳过")
            continue
        try:
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("width", 0))
            h = int(region.get("height", 0))
            _debug_log(f"[DEBUG] region[{i}] 解析: x={x}, y={y}, w={w}, h={h}")
        except (TypeError, ValueError) as e:
            _debug_log(f"[DEBUG] region[{i}] 解析失败: {e}")
            continue
        if w <= 0 or h <= 0:
            _debug_log(f"[DEBUG] region[{i}] w 或 h <= 0，跳过")
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        _debug_log(f"[DEBUG] region[{i}] 规范化后: x={x}, y={y}, w={w}, h={h}")
        normalized.append((x, y, w, h))
    _debug_log(f"[DEBUG] 最终 normalized: {normalized}")
    return normalized


def _swap_face_in_regions_for_frame(
    frame,
    normalized_regions,
    worker_tf,
    worker_lock,
    destination_face,
):
    """Swap faces in specified regions of a frame."""
    out = frame.copy()
    swapped_count = 0

    for x, y, w, h in normalized_regions:
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        with worker_lock:
            reference_face = worker_tf.get_one_face(crop)
        if reference_face is None:
            continue

        with worker_lock:
            swapped_crop = worker_tf.swap_face(
                vision_frame=crop,
                reference_face=reference_face,
                destination_face=destination_face,
            )
        if swapped_crop is None:
            continue

        out[y : y + h, x : x + w] = swapped_crop
        swapped_count += 1

    return out, swapped_count


def _normalize_regions_with_face_source(regions, width, height):
    """Normalize regions with face source info."""
    normalized = []
    if not regions:
        return normalized

    for region in regions:
        if not isinstance(region, dict):
            continue

        face_source_id = region.get("faceSourceId")
        if not face_source_id:
            continue

        try:
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("width", 0))
            h = int(region.get("height", 0))
        except (TypeError, ValueError):
            continue

        if w <= 0 or h <= 0:
            continue

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        normalized.append(
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "faceSourceId": str(face_source_id),
            }
        )

    return normalized


def _get_output_file_path(file_name):
    """Get the output file path."""
    base_name, ext = os.path.splitext(file_name)
    return base_name + "_output" + ext


def _get_output_video_path(file_name):
    """Get the output video path."""
    base_name, _ = os.path.splitext(file_name)
    return base_name + "_output.mp4"


@lru_cache(maxsize=1)
def _resolve_ffmpeg_binary() -> str | None:
    """优先解析 ffmpeg 可执行文件路径（支持打包目录兜底）。"""
    env_ffmpeg = os.environ.get("MAGIC_FFMPEG_PATH")
    if env_ffmpeg and os.path.exists(env_ffmpeg):
        return env_ffmpeg

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    exe_dir = os.path.dirname(os.path.abspath(sys.executable or ""))
    candidate_names = ["ffmpeg.exe", "ffmpeg"] if os.name == "nt" else ["ffmpeg"]
    for name in candidate_names:
        candidate = os.path.join(exe_dir, name)
        if os.path.exists(candidate):
            return candidate

    # 开发环境兜底：项目根目录下若放置了 ffmpeg
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    for name in candidate_names:
        candidate = os.path.join(base_dir, name)
        if os.path.exists(candidate):
            return candidate

    return None


@lru_cache(maxsize=1)
def _resolve_ffprobe_binary() -> str | None:
    """优先解析 ffprobe 可执行文件路径（支持与 ffmpeg 同目录）。"""
    env_ffprobe = os.environ.get("MAGIC_FFPROBE_PATH")
    if env_ffprobe and os.path.exists(env_ffprobe):
        return env_ffprobe

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    ffmpeg = _resolve_ffmpeg_binary()
    if ffmpeg:
        sibling = os.path.join(
            os.path.dirname(ffmpeg), "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if os.path.exists(sibling):
            return sibling

    exe_dir = os.path.dirname(os.path.abspath(sys.executable or ""))
    candidate_names = ["ffprobe.exe", "ffprobe"] if os.name == "nt" else ["ffprobe"]
    for name in candidate_names:
        candidate = os.path.join(exe_dir, name)
        if os.path.exists(candidate):
            return candidate

    # 开发环境兜底：项目根目录下若放置了 ffprobe
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    for name in candidate_names:
        candidate = os.path.join(base_dir, name)
        if os.path.exists(candidate):
            return candidate

    return None


def _resolve_total_frames(input_video_path: str, fps: float, current_total: int) -> int:
    """优先使用 OpenCV 帧数；若为 0，则尝试用 ffprobe 回退估算。"""
    if current_total and current_total > 0:
        return int(current_total)

    ffprobe = _resolve_ffprobe_binary()
    if not ffprobe:
        print("[WARN] 未找到 ffprobe，无法回退估算总帧数")
        return 0

    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration:format=duration",
            "-of",
            "json",
            input_video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            print("[WARN] ffprobe 获取总帧数失败，继续使用未知总帧数")
            return 0

        payload = json.loads(proc.stdout)
        stream = {}
        streams = payload.get("streams")
        if isinstance(streams, list) and streams:
            if isinstance(streams[0], dict):
                stream = streams[0]

        nb_frames_raw = stream.get("nb_frames")
        if nb_frames_raw not in (None, "", "N/A"):
            try:
                nb_frames = int(float(nb_frames_raw))
                if nb_frames > 0:
                    print(f"[INFO] ffprobe 检测总帧数: {nb_frames}")
                    return nb_frames
            except Exception:
                pass

        duration_raw = stream.get("duration")
        if duration_raw in (None, "", "N/A"):
            fmt = payload.get("format")
            if isinstance(fmt, dict):
                duration_raw = fmt.get("duration")

        if duration_raw not in (None, "", "N/A"):
            try:
                duration = float(duration_raw)
            except Exception:
                duration = 0.0
            if duration > 0 and fps and fps > 0:
                estimated = max(1, int(round(duration * fps)))
                print(f"[INFO] ffprobe 通过时长估算总帧数: {estimated}")
                return estimated

    except Exception as e:
        print(f"[WARN] ffprobe 回退估算总帧数异常: {str(e)}")

    return 0


def _input_has_audio_stream(input_video_path: str) -> bool | None:
    """检测输入视频是否包含音轨。True/False 表示可确定，None 表示无法判断。"""
    ffprobe = _resolve_ffprobe_binary()
    if not ffprobe:
        print("[WARN] 未找到 ffprobe，无法预检输入视频音轨")
        return None

    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            input_video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print("[WARN] ffprobe 预检音轨失败，将继续尝试复用音频")
            return None

        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams")
        if isinstance(streams, list):
            return len(streams) > 0
    except Exception as e:
        print(f"[WARN] ffprobe 预检音轨异常: {str(e)}")
    return None


def _mux_audio_or_raise(input_video_path: str, output_video_path: str):
    """原视频有音轨时，音频复用失败应明确报错；无音轨时允许跳过。"""
    has_audio = _input_has_audio_stream(input_video_path)
    if has_audio is False:
        print("[INFO] 原视频无音轨，跳过音频复用")
        return

    try:
        _try_mux_audio(input_video_path, output_video_path)
    except Exception as e:
        print(f"[ERROR] 音频复用失败: {str(e)}")
        raise RuntimeError("audio-mux-failed") from e


def _try_mux_audio(input_video_path: str, output_video_path: str):
    """使用 ffmpeg 将原视频音轨复用到输出视频（优先 copy，失败回退 aac 转码）。"""
    ffmpeg = _resolve_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg-not-found")

    tmp_path = os.path.splitext(output_video_path)[0] + "_mux_tmp.mp4"
    errors = []

    for audio_codec in ("copy", "aac"):
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            output_video_path,
            "-i",
            input_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            "-shortest",
            tmp_path,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            os.replace(tmp_path, output_video_path)
            return

        err_tail = (proc.stderr or "")[-320:]
        errors.append(f"{audio_codec}: {err_tail}")

        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    raise RuntimeError("ffmpeg failed: " + " | ".join(errors))


def _get_model_path(file_name: str):
    """Get the full path for a model file."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "models", file_name)
    )


def detect_face_boxes_in_image(input_path, regions=None):
    """Detect face bounding boxes in an image."""
    try:
        vision = _read_image(input_path)
        height, width = vision.shape[:2]

        # 优化：如果图片过大，先缩小进行检测，再映射回原坐标
        # 限制最大边长为 1920，既能保证检测精度，又能大幅提升速度
        max_size = 1920
        scale = 1.0
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_w = int(width * scale)
            new_h = int(height * scale)
            vision_resized = cv2.resize(vision, (new_w, new_h))
            print(f"[INFO] 图片过大 ({width}x{height})，缩放至 {new_w}x{new_h} 进行检测")

            # 缩放 regions
            search_areas_resized = []
            if regions:
                normalized = _normalize_regions(regions, width, height)
                for x, y, w, h in normalized:
                    search_areas_resized.append((
                        int(x * scale),
                        int(y * scale),
                        int(w * scale),
                        int(h * scale)
                    ))
            else:
                search_areas_resized = [(0, 0, new_w, new_h)]

            boxes_resized = _detect_face_boxes_in_frame(vision_resized, search_areas_resized, _tf, _tf_lock)

            # 映射回原图坐标
            boxes = []
            for bx, by, bw, bh in boxes_resized:
                boxes.append((
                    int(bx / scale),
                    int(by / scale),
                    int(bw / scale),
                    int(bh / scale)
                ))
        else:
            search_areas = (
                _normalize_regions(regions, width, height)
                if regions
                else [(0, 0, width, height)]
            )
            boxes = _detect_face_boxes_in_frame(vision, search_areas, _tf, _tf_lock)

        return [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in boxes]
    except Exception as e:
        _log_error("detect_face_boxes_in_image", e)
        raise


def detect_face_boxes_in_video(input_path, key_frame_ms=0, regions=None):
    """Detect face bounding boxes in video frames."""
    cap = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_index = 0
        if key_frame_ms and fps > 0:
            frame_index = int(round(max(0.0, float(key_frame_ms)) / 1000.0 * fps))
        if total_frames > 0:
            frame_index = max(0, min(frame_index, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("video-frame-read-failed")

        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]

        search_areas = (
            _normalize_regions(regions, width, height)
            if regions
            else [(0, 0, width, height)]
        )
        boxes = _detect_face_boxes_in_frame(frame, search_areas, _tf, _tf_lock)

        return {
            "regions": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in boxes],
            "frameWidth": width,
            "frameHeight": height,
            "frameIndex": frame_index,
        }
    except Exception as e:
        _log_error("detect_face_boxes_in_video", e)
        raise
    finally:
        if cap is not None:
            cap.release()


def swap_face_video_by_sources(
    input_path,
    face_sources,
    regions,
    key_frame_ms=0,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
):
    """Swap faces in a video using multiple face sources.

    Each region can use a different face source for replacement.
    Supports GPU acceleration and progress callbacks.
    """
    try:
        _emit_stage(stage_callback, "validating-input")
        if not os.path.exists(input_path):
            raise FileNotFoundError("file-not-found")

        save_path = _get_output_video_path(input_path)
        output_path = _swap_face_video_by_sources(
            input_path=input_path,
            face_sources=face_sources,
            regions=regions,
            key_frame_ms=key_frame_ms,
            save_path=save_path,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
        )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")

        _emit_stage(stage_callback, "muxing-audio")
        _mux_audio_or_raise(input_path, output_path)

        _emit_stage(stage_callback, "finalizing")
        return output_path
    except Exception as e:
        _log_error("swap_face_video_by_sources", e)
        raise


def _swap_face_video_by_sources(
    input_path,
    face_sources,
    regions,
    key_frame_ms,
    save_path,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
):
    """
    多人换脸视频处理（使用多线程架构）
    """
    cap = None
    writer = None

    # 关键修复：
    # 多人视频换脸依赖 tracks（跨帧状态）按时间顺序更新。
    # 若使用多处理线程并发，不同帧会乱序更新同一份 tracks，导致轨迹错配/丢失，
    # 表现为“选了多个人脸框，但有时只换其中一个”。
    # 因此这里固定为单处理线程，优先保证多人换脸正确性与稳定性。
    cpu_count = multiprocessing.cpu_count()
    num_workers = 1
    queue_size = 8 if use_gpu else 5

    print(
        f"[INFO] 多人换脸使用 {num_workers} 个处理线程（CPU核数={cpu_count}），队列大小: {queue_size}"
    )

    # 多线程队列
    read_queue = queue.Queue(maxsize=queue_size)
    write_queue = queue.PriorityQueue(maxsize=queue_size)

    # 控制标志
    stop_event = threading.Event()
    processing_error = threading.Lock()
    error_container = {'error': None}
    workers_done_event = threading.Event()
    workers_done_lock = threading.Lock()
    workers_done_count = {'count': 0}

    def _queue_put_with_stop(
        q_obj,
        item,
        *,
        timeout=1,
        warn_prefix="队列已满，等待中...",
    ) -> bool:
        """Put an item in a queue with stop signal support."""
        wait_count = 0
        while not stop_event.is_set():
            try:
                q_obj.put(item, timeout=timeout)
                return True
            except queue.Full:
                wait_count += 1
                if wait_count % 5 == 0:
                    print(f"[WARN] {warn_prefix} (已等待约 {wait_count} 秒)")
        return False

    def _mark_worker_done():
        """Mark a worker thread as done."""
        with workers_done_lock:
            workers_done_count['count'] += 1
            if workers_done_count['count'] >= num_workers:
                workers_done_event.set()

    try:
        _emit_stage(stage_callback, "opening-video")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")

        _emit_stage(stage_callback, "reading-video-metadata")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = _resolve_total_frames(input_path, fps, total_frames)

        if width <= 0 or height <= 0:
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                raise RuntimeError("video-open-failed")
            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        normalized_regions = _normalize_regions_with_face_source(regions, width, height)
        if not normalized_regions:
            raise RuntimeError("invalid-face-source-binding")

        _emit_stage(stage_callback, "extracting-target-face")

        gpu_pool_size = _resolve_gpu_pool_size(num_workers) if use_gpu else 1
        if use_gpu:
            _emit_stage(stage_callback, "gpu-initializing")
        tf_pool, using_gpu, selected_provider = _get_tf_pool(
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
            pool_size=gpu_pool_size,
        )
        if using_gpu:
            _emit_stage(stage_callback, "gpu-enabled")
            print(
                f"[INFO] 当前 GPU Provider: {selected_provider}, 实例池={len(tf_pool)}, worker={num_workers}"
            )
        elif use_gpu:
            _emit_stage(stage_callback, "gpu-fallback-cpu")
            print(f"[INFO] 已回退 CPU: worker={num_workers}")
        else:
            _emit_stage(stage_callback, "using-cpu")

        bootstrap_tf, bootstrap_lock = tf_pool[0]
        destination_faces = {}
        for source_id, source_path in face_sources.items():
            face_img = _read_image(source_path)
            with bootstrap_lock:
                destination_face = bootstrap_tf.get_one_face(face_img)
            if destination_face is None:
                raise RuntimeError("no-face-detected")
            destination_faces[str(source_id)] = destination_face

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("video-write-failed")

        key_frame_index = 0
        if key_frame_ms and fps > 0:
            key_frame_index = int(round(max(0.0, float(key_frame_ms)) / 1000.0 * fps))
        if total_frames > 0:
            key_frame_index = max(0, min(key_frame_index, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame_index)
        ok, key_frame = cap.read()
        if not ok or key_frame is None:
            raise RuntimeError("video-frame-read-failed")

        _emit_stage(stage_callback, "building-face-tracks")
        key_detections = _get_faces_with_boxes(key_frame, bootstrap_tf, bootstrap_lock)
        tracks = _build_tracks_from_seed_regions(normalized_regions, key_detections)
        if not tracks:
            raise RuntimeError("no-face-in-selected-regions")

        # 共享的轨迹数据（需要线程安全）
        tracks_lock = threading.Lock()
        stats = {
            'frame_count': 0,
            'processed_faces': 0,
            'start_time': time.time()
        }
        stats_lock = threading.Lock()

        # 读取线程
        def read_frames():
            """Read frames from the input video."""
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not _queue_put_with_stop(
                        read_queue,
                        (frame_idx, frame),
                        timeout=1,
                        warn_prefix="读取队列已满，等待处理线程消费",
                    ):
                        return
                    frame_idx += 1
                for _ in range(num_workers):
                    if not _queue_put_with_stop(
                        read_queue,
                        (None, None),
                        timeout=1,
                        warn_prefix="读取队列已满，等待投递结束信号",
                    ):
                        break
            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 读取线程异常: {str(e)}")
                stop_event.set()
                _clear_queue(read_queue)

        # 处理线程（多个）
        def process_frames(worker_id):
            """Process frames for face swapping."""
            worker_tf, worker_lock = tf_pool[worker_id % len(tf_pool)]
            try:
                _emit_stage(stage_callback, "processing-video-frames")
                while not stop_event.is_set():
                    try:
                        frame_idx, frame = read_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    if frame_idx is None:
                        break

                    with stats_lock:
                        stats['frame_count'] += 1
                        current_frame = stats['frame_count']

                    # 进度回调
                    if progress_callback and current_frame % 5 == 0:
                        try:
                            with stats_lock:
                                progress_callback(
                                    frame_count=current_frame,
                                    total_frames=total_frames,
                                    elapsed_seconds=max(0.0, time.time() - stats['start_time']),
                                )
                        except Exception as e:
                            print(f"[WARN] progress_callback failed: {str(e)}")

                    # 人脸检测
                    detections = _get_faces_with_boxes(frame, worker_tf, worker_lock)

                    # 匹配轨迹（需要锁保护）
                    with tracks_lock:
                        matches = _match_tracks_to_detections(tracks, detections)
                        matched_track_ids = set()

                        # 更新轨迹
                        for track_id, det_idx in matches:
                            track = tracks.get(track_id)
                            if track is None:
                                continue
                            detection = detections[det_idx]
                            track["box"] = detection["box"]
                            track["missed"] = 0
                            matched_track_ids.add(track_id)

                        # 清理过期轨迹
                        stale_track_ids = []
                        for track_id, track in tracks.items():
                            if track_id in matched_track_ids:
                                continue
                            track["missed"] = int(track.get("missed", 0)) + 1
                            # 容忍更长时间的短暂丢脸，避免某个目标被过早清理后不再参与换脸
                            if track["missed"] > 300:
                                stale_track_ids.append(track_id)

                        for track_id in stale_track_ids:
                            tracks.pop(track_id, None)

                    # 换脸处理
                    out = frame
                    for track_id, det_idx in matches:
                        with tracks_lock:
                            track = tracks.get(track_id)
                        if track is None:
                            continue

                        detection = detections[det_idx]
                        source_id = track.get("faceSourceId")
                        destination_face = destination_faces.get(str(source_id))
                        if destination_face is None:
                            continue

                        reference_face = detection.get("face")
                        if reference_face is None:
                            continue

                        try:
                            with worker_lock:
                                swapped = worker_tf.swap_face(
                                    vision_frame=out,
                                    reference_face=reference_face,
                                    destination_face=destination_face,
                                )
                            if swapped is not None:
                                out = swapped
                                with stats_lock:
                                    stats['processed_faces'] += 1
                        except Exception as e:
                            print(f"[WARN] 帧{current_frame} 轨迹{track_id} 换脸失败: {str(e)}")

                    out = _normalize_output_frame(out, width, height)
                    if not _queue_put_with_stop(
                        write_queue,
                        (frame_idx, out),
                        timeout=1,
                        warn_prefix=f"写入队列已满，Worker-{worker_id} 等待中",
                    ):
                        break

            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 处理线程 Worker-{worker_id} 异常: {str(e)}")
                stop_event.set()
            finally:
                _mark_worker_done()

        def write_frames():
            """Write processed frames to the output video."""
            try:
                next_frame_idx = 0
                frame_buffer = {}
                frames_written = 0

                while True:
                    if stop_event.is_set() and write_queue.empty():
                        break

                    try:
                        frame_idx, frame = write_queue.get(timeout=1)
                    except queue.Empty:
                        if workers_done_event.is_set():
                            if frame_buffer:
                                for pending_idx in sorted(frame_buffer.keys()):
                                    writer.write(frame_buffer[pending_idx])
                                    frames_written += 1
                                frame_buffer.clear()
                            break
                        continue

                    frame_buffer[frame_idx] = frame

                    while next_frame_idx in frame_buffer:
                        writer.write(frame_buffer.pop(next_frame_idx))
                        frames_written += 1
                        next_frame_idx += 1

                    if (
                        total_frames > 0
                        and frames_written >= total_frames
                        and workers_done_event.is_set()
                    ):
                        break

            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 写入线程异常: {str(e)}")
                stop_event.set()

        # 启动线程（移除 daemon=True 以确保线程正确完成）
        read_thread = threading.Thread(target=read_frames, name="VideoReader")
        process_threads = [
            threading.Thread(target=process_frames, args=(i,), name=f"VideoProcessor-{i}")
            for i in range(num_workers)
        ]
        write_thread = threading.Thread(target=write_frames, name="VideoWriter")

        read_thread.start()
        for t in process_threads:
            t.start()
        write_thread.start()

        # 等待所有线程完成
        read_thread.join()
        for t in process_threads:
            t.join()
        write_thread.join()

        # 检查是否有错误
        with processing_error:
            if error_container['error'] is not None:
                raise error_container['error']

        # 最终进度回调
        if progress_callback:
            try:
                final_total = total_frames if total_frames > 0 else stats['frame_count']
                progress_callback(
                    frame_count=final_total,
                    total_frames=final_total,
                    elapsed_seconds=max(0.0, time.time() - stats['start_time']),
                )
            except Exception as e:
                print(f"[WARN] progress_callback(final) failed: {str(e)}")

        print(
            f"[INFO] 视频多人换脸完成: 总帧={stats['frame_count']}, 成功换脸人次={stats['processed_faces']}, 轨迹数={len(tracks)}"
        )
        return save_path

    except Exception as e:
        stop_event.set()
        _log_error("_swap_face_video_by_sources", e)
        raise
    finally:
        stop_event.set()

        # 清空队列，避免线程阻塞
        _clear_queue(read_queue)
        _clear_queue(write_queue)

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def _normalize_output_frame(frame, width, height):
    """Normalize output frame dimensions."""
    out = frame
    if out is None:
        out = np.zeros((height, width, 3), dtype=np.uint8)
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif len(out.shape) == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    if out.shape[1] != width or out.shape[0] != height:
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
    if out.dtype != np.uint8:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out


def _detect_face_boxes_in_frame(frame, search_areas, tf_instance=None, tf_lock=None):
    """Detect face boxes in a frame."""
    frame_h, frame_w = frame.shape[:2]
    boxes = []
    for area in search_areas:
        x, y, w, h = area
        crop = frame[y : y + h, x : x + w]
        detections = _get_faces_with_boxes(crop, tf_instance, tf_lock)
        for det in detections:
            bx, by, bw, bh = det["box"]
            gx = x + bx
            gy = y + by
            sq = _expand_square_box(gx, gy, bw, bh, frame_w, frame_h)
            if sq is not None:
                boxes.append(sq)

    deduped = _dedupe_boxes(boxes, iou_threshold=0.45)
    deduped.sort(key=lambda b: (b[1], b[0]))
    return deduped


def _get_faces_with_boxes(frame, tf_instance=None, tf_lock=None):
    """获取帧中的人脸及其边界框"""
    if tf_instance is None:
        tf_instance = _tf
    if tf_lock is None:
        tf_lock = _tf_lock

    faces = []
    with tf_lock:
        if hasattr(tf_instance, "get_many_faces"):
            try:
                many = tf_instance.get_many_faces(frame)
                if many:
                    faces = list(many)
            except Exception:
                faces = []

        if not faces:
            try:
                one = tf_instance.get_one_face(frame)
                if one is not None:
                    faces = [one]
            except Exception:
                faces = []

    frame_h, frame_w = frame.shape[:2]
    out = []
    for face in faces:
        box = _extract_face_box(face, frame_w, frame_h)
        if box is None:
            continue
        out.append({"face": face, "box": box})
    return out


def _extract_face_box(face_obj, frame_w, frame_h):
    """Extract face box coordinates."""
    candidates = [
        face_obj,
        getattr(face_obj, "bbox", None),
        getattr(face_obj, "box", None),
        getattr(face_obj, "rect", None),
        getattr(face_obj, "bounding_box", None),
    ]

    for item in candidates:
        box = _parse_box_like(item, frame_w, frame_h)
        if box is not None:
            return box

    if isinstance(face_obj, dict):
        for key in ("bbox", "box", "rect", "bounding_box"):
            box = _parse_box_like(face_obj.get(key), frame_w, frame_h)
            if box is not None:
                return box

    return None


def _parse_box_like(raw, frame_w, frame_h):
    """Parse box-like coordinates from raw data."""
    if raw is None:
        return None

    if isinstance(raw, dict):
        if all(k in raw for k in ("x", "y", "width", "height")):
            x = _to_int(raw.get("x"))
            y = _to_int(raw.get("y"))
            w = _to_int(raw.get("width"))
            h = _to_int(raw.get("height"))
            return _clamp_box(x, y, w, h, frame_w, frame_h)
        if all(k in raw for k in ("x1", "y1", "x2", "y2")):
            x1 = _to_float(raw.get("x1"))
            y1 = _to_float(raw.get("y1"))
            x2 = _to_float(raw.get("x2"))
            y2 = _to_float(raw.get("y2"))
            return _from_xyxy(x1, y1, x2, y2, frame_w, frame_h)

    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 4:
        a = _to_float(raw[0])
        b = _to_float(raw[1])
        c = _to_float(raw[2])
        d = _to_float(raw[3])

        if max(abs(a), abs(b), abs(c), abs(d)) <= 2.0:
            a *= frame_w
            c *= frame_w
            b *= frame_h
            d *= frame_h

        if c > a and d > b:
            return _from_xyxy(a, b, c, d, frame_w, frame_h)

        return _clamp_box(_to_int(a), _to_int(b), _to_int(c), _to_int(d), frame_w, frame_h)

    # 对象字段尝试
    attrs = vars(raw) if hasattr(raw, "__dict__") else {}
    if attrs:
        return _parse_box_like(attrs, frame_w, frame_h)

    return None


def _from_xyxy(x1, y1, x2, y2, frame_w, frame_h):
    """Convert xyxy format to xywh format."""
    x = _to_int(min(x1, x2))
    y = _to_int(min(y1, y2))
    w = _to_int(abs(x2 - x1))
    h = _to_int(abs(y2 - y1))
    return _clamp_box(x, y, w, h, frame_w, frame_h)


def _clamp_box(x, y, w, h, frame_w, frame_h):
    """Clamp box coordinates to frame boundaries."""
    if frame_w <= 0 or frame_h <= 0:
        return None
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(int(x), frame_w - 1))
    y = max(0, min(int(y), frame_h - 1))
    w = max(1, min(int(w), frame_w - x))
    h = max(1, min(int(h), frame_h - y))
    return (x, y, w, h)


def _to_int(value):
    """Convert value to int safely."""
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _to_float(value):
    """Convert value to float safely."""
    try:
        return float(value)
    except Exception:
        return 0.0


def _expand_square_box(x, y, w, h, max_w, max_h, scale=1.35, min_size=48):
    """Expand a box to a square with given scale."""
    if w <= 0 or h <= 0:
        return None
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(float(w), float(h)) * float(scale)
    side = max(float(min_size), side)

    half = side / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    left = max(0, left)
    top = max(0, top)
    right = min(max_w, right)
    bottom = min(max_h, bottom)

    nw = right - left
    nh = bottom - top
    size = min(nw, nh)
    if size <= 2:
        return None

    # 再次强制为正方形
    right = left + size
    bottom = top + size
    return _clamp_box(left, top, size, size, max_w, max_h)


def _iou(box_a, box_b):
    """Calculate intersection over union of two boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a2x, b2x)
    inter_y2 = min(a2y, b2y)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _dedupe_boxes(boxes, iou_threshold=0.45):
    """Remove duplicate boxes based on IoU threshold."""
    out = []
    for box in boxes:
        keep = True
        for kept in out:
            if _iou(box, kept) >= iou_threshold:
                keep = False
                break
        if keep:
            out.append(box)
    return out


def _center_distance(box_a, box_b):
    """Calculate center distance between two boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _build_tracks_from_seed_regions(seed_regions, detections):
    """Build face tracks from seed regions."""
    if not seed_regions:
        return {}

    # 关键修复：
    # 即使关键帧检测不完整（例如只检测到 1 张脸），也要为每个用户选区创建轨迹，
    # 避免“两个框只初始化了一个轨迹”导致后续始终只换一个人。
    detections = detections or []

    tracks = {}
    used_det = set()
    track_id = 1

    for region in seed_regions:
        region_box = (region["x"], region["y"], region["width"], region["height"])
        best_idx = -1
        best_iou = 0.0

        for idx, det in enumerate(detections):
            if idx in used_det:
                continue
            iou = _iou(region_box, det["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx < 0:
            best_dist = None
            for idx, det in enumerate(detections):
                if idx in used_det:
                    continue
                dist = _center_distance(region_box, det["box"])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        # 若关键帧未匹配到检测框，回退使用用户选区本身作为初始轨迹框
        if best_idx >= 0:
            used_det.add(best_idx)
            init_box = detections[best_idx]["box"]
        else:
            init_box = region_box

        tracks[track_id] = {
            "trackId": track_id,
            "faceSourceId": str(region["faceSourceId"]),
            "box": init_box,
            "missed": 0,
        }
        track_id += 1

    return tracks


def _match_tracks_to_detections(tracks, detections):
    """Match existing tracks to new detections."""
    if not tracks or not detections:
        return []

    track_ids = list(tracks.keys())
    candidate_pairs = []

    for tid in track_ids:
        tbox = tracks[tid]["box"]
        for didx, det in enumerate(detections):
            iou = _iou(tbox, det["box"])
            if iou > 0.05:
                candidate_pairs.append((iou, tid, didx))

    candidate_pairs.sort(reverse=True, key=lambda item: item[0])

    matched_tracks = set()
    matched_dets = set()
    matches = []

    for score, tid, didx in candidate_pairs:
        if tid in matched_tracks or didx in matched_dets:
            continue
        matched_tracks.add(tid)
        matched_dets.add(didx)
        matches.append((tid, didx))

    # 对未匹配轨迹做一次基于中心点的兜底匹配
    for tid in track_ids:
        if tid in matched_tracks:
            continue
        tbox = tracks[tid]["box"]
        best_idx = -1
        best_dist = None
        for didx, det in enumerate(detections):
            if didx in matched_dets:
                continue
            dist = _center_distance(tbox, det["box"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = didx

        if best_idx >= 0:
            tw = max(1, tbox[2])
            th = max(1, tbox[3])
            max_dist = ((tw * tw + th * th) ** 0.5) * 0.65
            if best_dist is not None and best_dist <= max_dist:
                matched_tracks.add(tid)
                matched_dets.add(best_idx)
                matches.append((tid, best_idx))

    return matches


def swap_face_deep(input_path, face_paths, regions=None):
    """Perform deep face swap with multiple target faces."""
    try:
        if not isinstance(face_paths, list) or len(face_paths) == 0:
            raise RuntimeError("missing-params")

        save_path = _get_output_file_path(input_path)
        input_img = _read_image(input_path)
        height, width = input_img.shape[:2]

        destination_faces = _load_destination_faces(face_paths, _tf, _tf_lock)
        if not destination_faces:
            raise RuntimeError("no-face-detected")

        if regions:
            normalized_regions = _normalize_regions(regions, width, height)
            if not normalized_regions:
                raise RuntimeError("invalid-regions")
        else:
            normalized_regions = _sort_boxes_by_position(
                _detect_face_boxes_in_frame(
                    input_img,
                    [(0, 0, width, height)],
                    _tf,
                    _tf_lock,
                )
            )
            if not normalized_regions:
                raise RuntimeError("no-face-detected")

        output_img = input_img.copy()
        swapped_count = 0

        for index, (x, y, w, h) in enumerate(_sort_boxes_by_position(normalized_regions)):
            crop = input_img[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            with _tf_lock:
                reference_face = _tf.get_one_face(crop)
            if reference_face is None:
                continue

            destination_face = destination_faces[index % len(destination_faces)]
            with _tf_lock:
                output_crop = _tf.swap_face(
                    vision_frame=crop,
                    reference_face=reference_face,
                    destination_face=destination_face,
                )
            if output_crop is None:
                continue

            output_img[y : y + h, x : x + w] = output_crop
            swapped_count += 1

        if swapped_count == 0 and regions:
            return _write_image(save_path, output_img)
        if swapped_count == 0:
            raise RuntimeError("no-face-detected")
        return _write_image(save_path, output_img)

    except Exception as e:
        _log_error("swap_face_deep", e)
        raise


def swap_face_video_deep(
    input_path,
    face_paths,
    regions=None,
    key_frame_ms=0,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
    segment_duration_sec=12,
    segment_overlap_frames=6,
):
    """Deep swap faces in a video with advanced tracking.

    Uses temporal tracking and face re-identification for consistent results
    across video segments. Supports multiple face sources.
    """
    try:
        _emit_stage(stage_callback, "validating-input")
        if not os.path.exists(input_path):
            raise FileNotFoundError("file-not-found")
        if not isinstance(face_paths, list) or len(face_paths) == 0:
            raise RuntimeError("missing-params")

        for face_path in face_paths:
            if not isinstance(face_path, str) or not os.path.exists(face_path):
                raise FileNotFoundError("file-not-found")

        save_path = _get_output_video_path(input_path)
        output_path = _swap_face_video_deep(
            input_path=input_path,
            face_paths=face_paths,
            regions=regions,
            key_frame_ms=key_frame_ms,
            save_path=save_path,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_frames=segment_overlap_frames,
        )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")

        _emit_stage(stage_callback, "muxing-audio")
        _mux_audio_or_raise(input_path, output_path)

        _emit_stage(stage_callback, "finalizing")
        return output_path
    except Exception as e:
        _log_error("swap_face_video_deep", e)
        raise


def _swap_face_video_deep(
    input_path,
    face_paths,
    regions,
    key_frame_ms,
    save_path,
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
    gpu_provider="auto",
    segment_duration_sec=12,
    segment_overlap_frames=6,
):
    """Internal implementation for deep video face swapping."""
    cap = None
    temp_dir = None
    try:
        _emit_stage(stage_callback, "opening-video")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")

        _emit_stage(stage_callback, "reading-video-metadata")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = _resolve_total_frames(input_path, fps, total_frames)

        if width <= 0 or height <= 0:
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                raise RuntimeError("video-open-failed")
            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        normalized_regions = _normalize_regions(regions, width, height) if regions else []
        if regions and not normalized_regions:
            raise RuntimeError("invalid-regions")

        _emit_stage(stage_callback, "extracting-target-face")
        gpu_pool_size = _resolve_gpu_pool_size(1) if use_gpu else 1
        if use_gpu:
            _emit_stage(stage_callback, "gpu-initializing")
        tf_pool, using_gpu, selected_provider = _get_tf_pool(
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
            pool_size=gpu_pool_size,
        )
        if using_gpu:
            _emit_stage(stage_callback, "gpu-enabled")
            print(f"[INFO] 深度换脸启用 GPU Provider: {selected_provider}")
        elif use_gpu:
            _emit_stage(stage_callback, "gpu-fallback-cpu")
        else:
            _emit_stage(stage_callback, "using-cpu")

        bootstrap_tf, bootstrap_lock = tf_pool[0]
        destination_faces = _load_destination_faces(face_paths, bootstrap_tf, bootstrap_lock)
        if not destination_faces:
            raise RuntimeError("no-face-detected")

        if total_frames <= 0:
            raise RuntimeError("video-open-failed")

        segment_frames = max(1, int(round(max(1, int(segment_duration_sec or 12)) * fps)))
        overlap_frames = max(0, int(segment_overlap_frames or 0))
        segments = _plan_video_segments(total_frames, segment_frames, overlap_frames)

        temp_dir = tempfile.mkdtemp(prefix="magicmirror_deep_segments_")
        max_workers = min(
            len(segments),
            max(1, min(multiprocessing.cpu_count(), 4 if not use_gpu else 2)),
        )

        _emit_stage(stage_callback, "processing-video-segments")
        started_at = time.time()
        completed_frames = 0
        segment_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {
                executor.submit(
                    _process_deep_video_segment,
                    input_path=input_path,
                    segment=segment,
                    destination_faces=destination_faces,
                    normalized_regions=normalized_regions,
                    fps=fps,
                    width=width,
                    height=height,
                    temp_dir=temp_dir,
                    use_gpu=use_gpu,
                    gpu_provider=gpu_provider,
                ): segment
                for segment in segments
            }

            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                result = future.result()
                segment_results[segment["index"]] = result
                completed_frames += int(result.get("framesWritten", 0) or 0)
                if progress_callback:
                    try:
                        progress_callback(
                            frame_count=min(completed_frames, total_frames),
                            total_frames=total_frames,
                            elapsed_seconds=max(0.0, time.time() - started_at),
                        )
                    except Exception as e:
                        print(f"[WARN] progress_callback failed: {str(e)}")

        ordered_segment_paths = [
            segment_results[index]["path"]
            for index in sorted(segment_results.keys())
            if segment_results.get(index) and segment_results[index].get("path")
        ]
        if not ordered_segment_paths:
            raise RuntimeError("video-output-missing")

        _emit_stage(stage_callback, "merging-video-segments")
        _concat_video_segments(ordered_segment_paths, save_path, fps, width, height)
        return save_path

    except Exception as e:
        _log_error("_swap_face_video_deep", e)
        raise
    finally:
        if cap is not None:
            cap.release()
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _load_destination_faces(face_paths, tf_instance=None, tf_lock=None):
    """Load destination face embeddings."""
    if tf_instance is None:
        tf_instance = _tf
    if tf_lock is None:
        tf_lock = _tf_lock

    destination_faces = []
    for face_path in face_paths:
        if isinstance(face_path, dict):
            face_path = face_path.get("path")
        if not isinstance(face_path, str) or not face_path:
            continue
        face_img = _read_image(face_path)
        with tf_lock:
            destination_face = tf_instance.get_one_face(face_img)
        if destination_face is None:
            print(f"[WARN] 目标脸素材未检测到人脸，已跳过: {face_path}")
            continue
        destination_faces.append(destination_face)
    return destination_faces


def _sort_boxes_by_position(boxes):
    """Sort boxes by position."""
    return sorted(boxes or [], key=lambda item: (int(item[1]), int(item[0])))


def _sort_detections_by_position(detections):
    """Sort detections by position."""
    return sorted(
        detections or [],
        key=lambda item: (
            int(item.get("box", (0, 0, 0, 0))[1]),
            int(item.get("box", (0, 0, 0, 0))[0]),
        ),
    )


def _build_deep_tracks_from_seed_regions(seed_regions, detections, target_count):
    """Build deep tracks from seed regions."""
    tracks = {}
    used_det = set()
    track_id = 1
    sorted_regions = _sort_boxes_by_position(seed_regions)

    for index, region_box in enumerate(sorted_regions):
        best_idx = -1
        best_iou = 0.0

        for det_idx, det in enumerate(detections or []):
            if det_idx in used_det:
                continue
            iou = _iou(region_box, det["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = det_idx

        if best_idx < 0:
            best_dist = None
            for det_idx, det in enumerate(detections or []):
                if det_idx in used_det:
                    continue
                dist = _center_distance(region_box, det["box"])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = det_idx

        if best_idx >= 0:
            used_det.add(best_idx)
            init_box = detections[best_idx]["box"]
        else:
            init_box = region_box

        tracks[track_id] = {
            "trackId": track_id,
            "targetIndex": index % max(1, int(target_count or 1)),
            "box": init_box,
            "missed": 0,
        }
        track_id += 1

    return tracks


def _build_deep_tracks_from_detections(detections, target_count):
    """Build deep tracks from detections."""
    tracks = {}
    sorted_detections = _sort_detections_by_position(detections)
    for index, det in enumerate(sorted_detections, start=1):
        tracks[index] = {
            "trackId": index,
            "targetIndex": (index - 1) % max(1, int(target_count or 1)),
            "box": det["box"],
            "missed": 0,
        }
    return tracks


def _plan_video_segments(total_frames, segment_frames, overlap_frames):
    """Plan video segments for processing."""
    segments = []
    if total_frames <= 0:
        return segments

    segment_frames = max(1, int(segment_frames or 1))
    overlap_frames = max(0, int(overlap_frames or 0))

    core_start = 0
    index = 0
    while core_start < total_frames:
        core_end = min(total_frames, core_start + segment_frames)
        read_start = max(0, core_start - overlap_frames)
        read_end = min(total_frames, core_end + overlap_frames)
        segments.append(
            {
                "index": index,
                "coreStart": core_start,
                "coreEnd": core_end,
                "readStart": read_start,
                "readEnd": read_end,
            }
        )
        index += 1
        core_start = core_end
    return segments


def _process_deep_video_segment(
    input_path,
    segment,
    destination_faces,
    normalized_regions,
    fps,
    width,
    height,
    temp_dir,
    use_gpu=False,
    gpu_provider="auto",
):
    """Process a single deep video segment."""
    cap = None
    writer = None
    try:
        segment_index = int(segment["index"])
        core_start = int(segment["coreStart"])
        core_end = int(segment["coreEnd"])
        read_start = int(segment["readStart"])
        read_end = int(segment["readEnd"])

        temp_output_path = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")
        cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("video-write-failed")

        tf_pool, _, _ = _get_tf_pool(
            use_gpu=use_gpu,
            gpu_provider=gpu_provider,
            pool_size=1,
        )
        worker_tf, worker_lock = tf_pool[0]

        tracks = {}
        next_track_id = 1
        frames_written = 0
        track_missed_limit = 90

        frame_index = read_start
        while frame_index < read_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            detections = _sort_detections_by_position(
                _get_faces_with_boxes(frame, worker_tf, worker_lock)
            )

            if not tracks:
                if normalized_regions:
                    tracks = _build_deep_tracks_from_seed_regions(
                        normalized_regions,
                        detections,
                        len(destination_faces),
                    )
                else:
                    tracks = _build_deep_tracks_from_detections(
                        detections,
                        len(destination_faces),
                    )
                next_track_id = max(tracks.keys(), default=0) + 1

            matches = _match_tracks_to_detections(tracks, detections) if tracks else []
            matched_track_ids = set()
            matched_detection_ids = set()

            for track_id, detection_index in matches:
                track = tracks.get(track_id)
                if track is None:
                    continue
                track["box"] = detections[detection_index]["box"]
                track["missed"] = 0
                matched_track_ids.add(track_id)
                matched_detection_ids.add(detection_index)

            stale_track_ids = []
            for track_id, track in tracks.items():
                if track_id in matched_track_ids:
                    continue
                track["missed"] = int(track.get("missed", 0)) + 1
                if int(track["missed"]) > track_missed_limit:
                    stale_track_ids.append(track_id)

            for track_id in stale_track_ids:
                tracks.pop(track_id, None)

            new_matches = []
            if not normalized_regions:
                for detection_index, detection in enumerate(detections):
                    if detection_index in matched_detection_ids:
                        continue
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {
                        "trackId": track_id,
                        "targetIndex": (track_id - 1) % max(1, len(destination_faces)),
                        "box": detection["box"],
                        "missed": 0,
                    }
                    new_matches.append((track_id, detection_index))

            out = frame
            for track_id, detection_index in matches + new_matches:
                track = tracks.get(track_id)
                if track is None:
                    continue
                reference_face = detections[detection_index].get("face")
                if reference_face is None:
                    continue
                destination_face = destination_faces[
                    int(track.get("targetIndex", 0)) % len(destination_faces)
                ]
                with worker_lock:
                    swapped = worker_tf.swap_face(
                        vision_frame=out,
                        reference_face=reference_face,
                        destination_face=destination_face,
                    )
                if swapped is not None:
                    out = swapped

            out = _normalize_output_frame(out, width, height)
            if core_start <= frame_index < core_end:
                writer.write(out)
                frames_written += 1

            frame_index += 1

        return {"path": temp_output_path, "framesWritten": frames_written}

    except Exception as e:
        _log_error("_process_deep_video_segment", e)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def _concat_video_segments(segment_paths, output_path, fps, width, height):
    """Concatenate video segments into final output."""
    writer = None
    caps = []
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("video-write-failed")

        for segment_path in segment_paths:
            cap = cv2.VideoCapture(segment_path)
            caps.append(cap)
            if not cap.isOpened():
                raise RuntimeError("video-open-failed")

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(_normalize_output_frame(frame, width, height))

        if not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")
        return output_path
    except Exception as e:
        _log_error("_concat_video_segments", e)
        raise
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            writer.release()
