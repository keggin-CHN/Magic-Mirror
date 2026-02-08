import os
import shutil
import subprocess
import threading
import traceback
from functools import lru_cache
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import cv2
import numpy as np
from tinyface import TinyFace

# 全局 TinyFace 实例（CPU 版本）
_tf = TinyFace()
_tf_lock = threading.RLock()

# GPU 加速的 TinyFace 实例（按需创建）
_tf_gpu = None
_tf_gpu_lock = threading.RLock()
_gpu_initialized = False


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
    try:
        _tf.config.face_detector_model = _get_model_path("scrfd_2.5g.onnx")
        _tf.config.face_embedder_model = _get_model_path("arcface_w600k_r50.onnx")
        _tf.config.face_swapper_model = _get_model_path("inswapper_128_fp16.onnx")
        _tf.config.face_enhancer_model = _get_model_path("gfpgan_1.4.onnx")
        _tf.prepare()
        return True
    except BaseException as _:
        return False


def _init_gpu_models():
    """初始化 GPU 加速的模型（使用 DirectML）- 双重检查锁定模式"""
    global _tf_gpu, _gpu_initialized
    
    # 第一次检查（无锁，快速路径）
    if _gpu_initialized:
        return _tf_gpu is not None
    
    # 使用锁保护初始化检查，避免竞态条件
    with _tf_gpu_lock:
        # 第二次检查（有锁，确保只初始化一次）
        if _gpu_initialized:
            return _tf_gpu is not None
        
        try:
            print("[INFO] 正在初始化 GPU 加速模型...")
            import onnxruntime as ort
            
            # 检查可用的 ExecutionProvider
            available_providers = ort.get_available_providers()
            print(f"[INFO] 可用的 ExecutionProvider: {available_providers}")
            
            # 优先使用 DirectML（Windows 通用），其次 CUDA
            if 'DmlExecutionProvider' in available_providers:
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                print("[INFO] 使用 DirectML 加速")
            elif 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("[INFO] 使用 CUDA 加速")
            else:
                print("[WARN] 未找到 GPU ExecutionProvider，回退到 CPU")
                _gpu_initialized = True
                return False
            
            # 创建 GPU 版本的 TinyFace 实例
            _tf_gpu = TinyFace()
            _tf_gpu.config.face_detector_model = _get_model_path("scrfd_2.5g.onnx")
            _tf_gpu.config.face_embedder_model = _get_model_path("arcface_w600k_r50.onnx")
            _tf_gpu.config.face_swapper_model = _get_model_path("inswapper_128_fp16.onnx")
            _tf_gpu.config.face_enhancer_model = _get_model_path("gfpgan_1.4.onnx")
            
            # 设置 ExecutionProvider
            _tf_gpu.config.execution_providers = providers
            
            _tf_gpu.prepare()
            _gpu_initialized = True
            print("[SUCCESS] GPU 模型初始化成功")
            return True
            
        except Exception as e:
            print(f"[ERROR] GPU 模型初始化失败: {str(e)}")
            print(traceback.format_exc())
            _tf_gpu = None
            _gpu_initialized = True
            return False


def _get_tf_instance(use_gpu=False):
    """获取 TinyFace 实例（CPU 或 GPU）"""
    if use_gpu:
        if _init_gpu_models() and _tf_gpu is not None:
            return _tf_gpu, _tf_gpu_lock
        else:
            print("[WARN] GPU 不可用，回退到 CPU")
    return _tf, _tf_lock


def _emit_stage(stage_callback, stage: str):
    if stage_callback is None:
        return
    try:
        stage_callback(stage)
    except Exception as e:
        print(f"[WARN] stage_callback failed: {str(e)}")


def swap_face(input_path, face_path):
    save_path = _get_output_file_path(input_path)
    output_img = _swap_face(input_path, face_path)
    return _write_image(save_path, output_img)


def swap_face_regions(input_path, face_path, regions):
    try:
        print(f"[DEBUG] swap_face_regions 被调用")
        print(f"[DEBUG] input_path: {input_path}")
        print(f"[DEBUG] face_path: {face_path}")
        print(f"[DEBUG] regions 类型: {type(regions)}, 值: {regions}")
        
        save_path = _get_output_file_path(input_path)
        input_img = _read_image(input_path)
        height, width = input_img.shape[:2]
        print(f"[DEBUG] 图片尺寸: {width}x{height}")
        
        normalized_regions = _normalize_regions(regions, width, height)
        print(f"[DEBUG] normalized_regions: {normalized_regions}")

        # 未选择/无有效选区：回退全图换脸
        if not normalized_regions:
            print("[WARN] 无有效选区，回退全图换脸！")
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


def swap_face_video(input_path, face_path, progress_callback=None, stage_callback=None, use_gpu=False):
    try:
        _emit_stage(stage_callback, "validating-input")
        print(f"[INFO] 开始视频换脸: input={input_path}, face={face_path}, use_gpu={use_gpu}")

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
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            use_gpu=use_gpu,
        )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")

        # 尝试使用 ffmpeg 把原视频音频复用到输出（OpenCV 写入的视频默认没有音轨）
        _emit_stage(stage_callback, "muxing-audio")
        try:
            _try_mux_audio(input_path, output_path)
        except Exception as e:
            print(f"[WARN] 音频复用失败，将返回无音轨视频: {str(e)}")

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
    progress_callback=None,
    stage_callback=None,
    use_gpu=False,
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
    
    # 动态计算队列大小和线程数
    cpu_count = multiprocessing.cpu_count()
    # GPU模式：使用较少线程避免锁竞争；CPU模式：使用更多线程
    # 确保至少有1个worker，最多不超过8个
    if use_gpu:
        num_workers = 2
    else:
        num_workers = max(1, min(cpu_count - 1, 8))
    queue_size = max(5, num_workers * 2)  # 队列大小为线程数的2倍
    
    print(f"[INFO] 使用 {num_workers} 个处理线程，队列大小: {queue_size}")
    
    # 多线程队列
    read_queue = queue.Queue(maxsize=queue_size)
    write_queue = queue.PriorityQueue(maxsize=queue_size)  # 使用优先队列保证顺序
    
    # 控制标志
    stop_event = threading.Event()  # 统一的停止标志
    processing_error = threading.Lock()  # 使用锁保护错误
    error_container = {'error': None}  # 线程安全的错误容器

    try:
        _emit_stage(stage_callback, "opening-video")
        print(f"[INFO] 打开视频文件: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("video-open-failed")

        # 获取视频属性
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

        print(f"[INFO] 视频尺寸: {width}x{height}, 总帧数: {total_frames}")

        if width <= 0 or height <= 0:
            print("[WARN] 无法获取视频尺寸，尝试读取第一帧")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("video-open-failed")
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(f"[INFO] 从第一帧获取尺寸: {width}x{height}")

        # 创建视频写入器
        print(f"[INFO] 创建输出视频: {save_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("video-write-failed")

        # 提取目标人脸
        _emit_stage(stage_callback, "extracting-target-face")
        print(f"[INFO] 提取目标人脸: {face_path}")
        tf_instance, tf_lock = _get_tf_instance(use_gpu)
        with tf_lock:
            destination_face = tf_instance.get_one_face(_read_image(face_path))
        if destination_face is None:
            raise RuntimeError("no-face-detected")
        print("[SUCCESS] 成功提取目标人脸")

        # 统计信息
        stats = {
            'frame_count': 0,
            'processed_count': 0,
            'failed_count': 0,
            'start_time': time.time()
        }
        stats_lock = threading.Lock()

        # 读取线程
        def read_frames():
            try:
                frame_idx = 0
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # 添加超时和重试机制，避免队列满时死锁
                    retry_count = 0
                    while not stop_event.is_set() and retry_count < 5:
                        try:
                            read_queue.put((frame_idx, frame), timeout=2)
                            break
                        except queue.Full:
                            retry_count += 1
                            print(f"[WARN] 读取队列已满，等待中... (重试 {retry_count}/5)")
                            if retry_count >= 5:
                                print("[ERROR] 读取队列持续满载，可能发生死锁")
                                stop_event.set()
                                # 清空队列避免死锁
                                _clear_queue(read_queue)
                                return
                    frame_idx += 1
                # 发送结束信号
                for _ in range(num_workers):
                    try:
                        read_queue.put((None, None), timeout=5)
                    except queue.Full:
                        print("[WARN] 无法发送结束信号到读取队列")
            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 读取线程异常: {str(e)}")
                stop_event.set()
                # 清空队列避免死锁
                _clear_queue(read_queue)

        # 处理线程（多个）
        def process_frames(worker_id):
            try:
                while not stop_event.is_set():
                    try:
                        frame_idx, frame = read_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # 结束信号
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
                    
                    if current_frame % 30 == 0:
                        progress = (current_frame / total_frames * 100) if total_frames > 0 else 0
                        print(f"[PROGRESS] 处理进度: {current_frame}/{total_frames} ({progress:.1f}%) [Worker-{worker_id}]")
                    
                    # 人脸检测和换脸
                    try:
                        with tf_lock:
                            reference_face = tf_instance.get_one_face(frame)
                        
                        if reference_face is None:
                            write_queue.put((frame_idx, frame))
                            with stats_lock:
                                stats['failed_count'] += 1
                            continue
                        
                        with tf_lock:
                            output_frame = tf_instance.swap_face(
                                vision_frame=frame,
                                reference_face=reference_face,
                                destination_face=destination_face,
                            )
                        
                        out = output_frame if output_frame is not None else frame
                        out = _normalize_output_frame(out, width, height)
                        write_queue.put((frame_idx, out))
                        
                        with stats_lock:
                            stats['processed_count'] += 1
                    
                    except Exception as e:
                        print(f"[WARN] 第{current_frame}帧处理失败: {str(e)}")
                        write_queue.put((frame_idx, frame))
                        with stats_lock:
                            stats['failed_count'] += 1
            
            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 处理线程 Worker-{worker_id} 异常: {str(e)}")
                stop_event.set()

        # 写入线程（按顺序写入）
        def write_frames():
            try:
                _emit_stage(stage_callback, "processing-video-frames")
                next_frame_idx = 0
                frame_buffer = {}  # 缓存乱序到达的帧
                frames_written = 0
                idle_count = 0  # 空闲计数器
                max_idle = 30  # 最多等待30秒（30次 * 1秒）
                
                while not stop_event.is_set():
                    try:
                        frame_idx, frame = write_queue.get(timeout=1)
                        idle_count = 0  # 重置空闲计数
                    except queue.Empty:
                        idle_count += 1
                        # 如果已经写入了所有帧，或者空闲时间过长，退出
                        if (total_frames > 0 and frames_written >= total_frames) or idle_count >= max_idle:
                            break
                        continue
                    
                    # 缓存帧
                    frame_buffer[frame_idx] = frame
                    
                    # 按顺序写入
                    while next_frame_idx in frame_buffer:
                        writer.write(frame_buffer.pop(next_frame_idx))
                        frames_written += 1
                        next_frame_idx += 1
                    
                    # 检查是否完成（但不立即退出，等待缓冲区中的帧）
                    if total_frames > 0 and frames_written >= total_frames and not frame_buffer:
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
        
        print("[INFO] 视频处理完成:")
        print(f"  - 总帧数: {stats['frame_count']}")
        print(f"  - 成功换脸: {stats['processed_count']}")
        print(f"  - 跳过/失败: {stats['failed_count']}")
        
        # 最终进度回调
        if progress_callback:
            try:
                final_count = total_frames if total_frames > 0 else stats['frame_count']
                progress_callback(
                    frame_count=final_count,
                    total_frames=final_count,
                    elapsed_seconds=max(0.0, time.time() - stats['start_time']),
                )
            except Exception as e:
                print(f"[WARN] progress_callback(final) failed: {str(e)}")

        return save_path

    except Exception as e:
        # 停止所有线程
        stop_event.set()
        _log_error("_swap_face_video", e)
        raise

    finally:
        # 确保所有线程停止
        stop_event.set()
        
        # 清空队列，避免线程阻塞
        _clear_queue(read_queue)
        _clear_queue(write_queue)
        
        if cap is not None:
            cap.release()
            print("[INFO] 释放视频读取器")
        if writer is not None:
            writer.release()
            print("[INFO] 释放视频写入器")


def _swap_face(input_path, face_path):
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
    face_img = _read_image(face_path)
    with _tf_lock:
        return _tf.get_one_face(face_img)


def _read_image(img_path: str):
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
    if img is None:
        raise RuntimeError("swap-failed")

    suffix = (os.path.splitext(img_path)[-1] or ".png").lower()

    def _try_write(path: str, ext: str) -> bool:
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
    normalized = []
    print(f"[DEBUG] _normalize_regions: regions={regions}, 图片尺寸={width}x{height}")
    if not regions:
        print("[DEBUG] regions 为空或 None")
        return normalized
    for i, region in enumerate(regions):
        print(f"[DEBUG] 处理 region[{i}]: type={type(region)}, value={region}")
        if not isinstance(region, dict):
            print(f"[DEBUG] region[{i}] 不是 dict，跳过")
            continue
        try:
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("width", 0))
            h = int(region.get("height", 0))
            print(f"[DEBUG] region[{i}] 解析: x={x}, y={y}, w={w}, h={h}")
        except (TypeError, ValueError) as e:
            print(f"[DEBUG] region[{i}] 解析失败: {e}")
            continue
        if w <= 0 or h <= 0:
            print(f"[DEBUG] region[{i}] w 或 h <= 0，跳过")
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        print(f"[DEBUG] region[{i}] 规范化后: x={x}, y={y}, w={w}, h={h}")
        normalized.append((x, y, w, h))
    print(f"[DEBUG] 最终 normalized: {normalized}")
    return normalized


def _normalize_regions_with_face_source(regions, width, height):
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
    base_name, ext = os.path.splitext(file_name)
    return base_name + "_output" + ext


def _get_output_video_path(file_name):
    base_name, _ = os.path.splitext(file_name)
    return base_name + "_output.mp4"


def _try_mux_audio(input_video_path: str, output_video_path: str):
    """如果系统中存在 ffmpeg，尝试把原视频音频复用到输出视频中（失败则忽略）。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return

    tmp_path = os.path.splitext(output_video_path)[0] + "_mux_tmp.mp4"
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
        "aac",
        "-shortest",
        tmp_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")

    os.replace(tmp_path, output_video_path)


def _get_model_path(file_name: str):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "models", file_name)
    )


def detect_face_boxes_in_image(input_path, regions=None):
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
):
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
        )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("video-output-missing")

        _emit_stage(stage_callback, "muxing-audio")
        try:
            _try_mux_audio(input_path, output_path)
        except Exception as e:
            print(f"[WARN] 音频复用失败，将返回无音轨视频: {str(e)}")

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
):
    """
    多人换脸视频处理（使用多线程架构）
    """
    cap = None
    writer = None
    
    # 动态计算队列大小和线程数
    cpu_count = multiprocessing.cpu_count()
    # 确保至少有1个worker，最多不超过8个
    if use_gpu:
        num_workers = 2
    else:
        num_workers = max(1, min(cpu_count - 1, 8))
    queue_size = max(5, num_workers * 2)
    
    print(f"[INFO] 多人换脸使用 {num_workers} 个处理线程，队列大小: {queue_size}")
    
    # 多线程队列
    read_queue = queue.Queue(maxsize=queue_size)
    write_queue = queue.PriorityQueue(maxsize=queue_size)
    
    # 控制标志
    stop_event = threading.Event()
    processing_error = threading.Lock()
    error_container = {'error': None}
    
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
        tf_instance, tf_lock = _get_tf_instance(use_gpu)
        destination_faces = {}
        for source_id, source_path in face_sources.items():
            face_img = _read_image(source_path)
            with tf_lock:
                destination_face = tf_instance.get_one_face(face_img)
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
        key_detections = _get_faces_with_boxes(key_frame, tf_instance, tf_lock)
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
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # 添加超时和重试机制，避免队列满时死锁
                    retry_count = 0
                    while not stop_event.is_set() and retry_count < 5:
                        try:
                            read_queue.put((frame_idx, frame), timeout=2)
                            break
                        except queue.Full:
                            retry_count += 1
                            print(f"[WARN] 读取队列已满，等待中... (重试 {retry_count}/5)")
                            if retry_count >= 5:
                                print("[ERROR] 读取队列持续满载，可能发生死锁")
                                stop_event.set()
                                _clear_queue(read_queue)
                                return
                    frame_idx += 1
                for _ in range(num_workers):
                    try:
                        read_queue.put((None, None), timeout=5)
                    except queue.Full:
                        print("[WARN] 无法发送结束信号到读取队列")
            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 读取线程异常: {str(e)}")
                stop_event.set()
                _clear_queue(read_queue)

        # 处理线程（多个）
        def process_frames(worker_id):
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
                    detections = _get_faces_with_boxes(frame, tf_instance, tf_lock)
                    
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
                            if track["missed"] > 45:
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
                            with tf_lock:
                                swapped = tf_instance.swap_face(
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
                    write_queue.put((frame_idx, out))
                    
            except Exception as e:
                with processing_error:
                    error_container['error'] = e
                print(f"[ERROR] 处理线程 Worker-{worker_id} 异常: {str(e)}")
                stop_event.set()

        # 写入线程
        def write_frames():
            try:
                next_frame_idx = 0
                frame_buffer = {}
                frames_written = 0
                idle_count = 0
                max_idle = 30
                
                while not stop_event.is_set():
                    try:
                        frame_idx, frame = write_queue.get(timeout=1)
                        idle_count = 0
                    except queue.Empty:
                        idle_count += 1
                        if (total_frames > 0 and frames_written >= total_frames) or idle_count >= max_idle:
                            break
                        continue
                    
                    frame_buffer[frame_idx] = frame
                    
                    while next_frame_idx in frame_buffer:
                        writer.write(frame_buffer.pop(next_frame_idx))
                        frames_written += 1
                        next_frame_idx += 1
                    
                    if total_frames > 0 and frames_written >= total_frames and not frame_buffer:
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
    x = _to_int(min(x1, x2))
    y = _to_int(min(y1, y2))
    w = _to_int(abs(x2 - x1))
    h = _to_int(abs(y2 - y1))
    return _clamp_box(x, y, w, h, frame_w, frame_h)


def _clamp_box(x, y, w, h, frame_w, frame_h):
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
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _expand_square_box(x, y, w, h, max_w, max_h, scale=1.35, min_size=48):
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
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _build_tracks_from_seed_regions(seed_regions, detections):
    if not seed_regions or not detections:
        return {}

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

        if best_idx < 0:
            continue

        used_det.add(best_idx)
        tracks[track_id] = {
            "trackId": track_id,
            "faceSourceId": str(region["faceSourceId"]),
            "box": detections[best_idx]["box"],
            "missed": 0,
        }
        track_id += 1

    return tracks


def _match_tracks_to_detections(tracks, detections):
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
