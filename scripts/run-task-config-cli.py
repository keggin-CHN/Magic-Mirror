#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON_DIR = PROJECT_ROOT / "src-python"
if str(SRC_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON_DIR))

from magic.face import (  # noqa: E402
    load_models,
    swap_face_video,
    swap_face_video_by_sources,
    swap_face_video_deep,
)
from magic.task_config import (  # noqa: E402
    get_expected_face_source_sha256_map,
    get_expected_input_video_sha256,
    get_expected_target_face_sha256,
    get_expected_target_faces_sha256_map,
    parse_video_task_config_token,
    verify_file_sha256,
)

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

VIDEO_TASK_CONFIG_TTL_SECONDS = int(
    os.environ.get("VIDEO_TASK_CONFIG_TTL_SECONDS", str(7 * 24 * 3600))
)
VIDEO_TASK_CONFIG_SECRET = os.environ.get(
    "VIDEO_TASK_CONFIG_SECRET", "magic-mirror-config-secret"
)


def _parse_video_task_config_token(config_id: str):
    return parse_video_task_config_token(
        str(config_id),
        VIDEO_TASK_CONFIG_SECRET,
        legacy_ttl_seconds=VIDEO_TASK_CONFIG_TTL_SECONDS,
    )


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _validate_file(path: str, allowed_exts: set[str], *, missing_code: str):
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
        "audio-mux-failed",
        "video-frame-read-failed",
        "output-write-failed",
        "invalid-regions",
        "config-mismatch",
        "config-not-found",
        "model-load-failed",
    ]
    for code in codes:
        if code in msg:
            return code
    return "internal"


def _normalize_gpu_provider(gpu_provider: str):
    mode = (gpu_provider or "auto").strip().lower()
    if mode in {"dml", "directml"}:
        return "directml"
    if mode == "cuda":
        return "cuda"
    if mode == "cpu":
        return "cpu"
    return "auto"


def _parse_face_source_entry(raw: str):
    if "=" not in raw:
        raise RuntimeError("invalid-face-source-binding")
    source_id, source_path = raw.split("=", 1)
    source_id = str(source_id).strip()
    source_path = source_path.strip()
    if not source_id or not source_path:
        raise RuntimeError("invalid-face-source-binding")
    return source_id, source_path


def _load_library_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}

    if isinstance(payload, list):
        out = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_id = item.get("id")
            source_path = item.get("path")
            if source_id is None or not source_path:
                continue
            out[str(source_id)] = str(source_path)
        return out

    raise RuntimeError("invalid-face-source-binding")


def _build_face_source_override_map(args):
    source_map = {}
    if args.library_map_json:
        source_map.update(_load_library_map(args.library_map_json))

    for item in args.face_source or []:
        source_id, source_path = _parse_face_source_entry(item)
        source_map[str(source_id)] = source_path

    return source_map


def _parse_target_face_item_entry(raw: str):
    if "=" not in raw:
        raise RuntimeError("invalid-face-source-binding")
    target_id, target_path = raw.split("=", 1)
    target_id = str(target_id).strip()
    target_path = target_path.strip()
    if not target_id or not target_path:
        raise RuntimeError("invalid-face-source-binding")
    return target_id, target_path


def _build_target_face_override_map(args):
    target_map = {}
    if args.library_map_json:
        target_map.update(_load_library_map(args.library_map_json))

    for item in args.target_face_item or []:
        target_id, target_path = _parse_target_face_item_entry(item)
        target_map[str(target_id)] = target_path

    return target_map


def _resolve_face_source_map(face_sources, override_map):
    if not isinstance(face_sources, list) or len(face_sources) == 0:
        raise RuntimeError("missing-face-sources")

    source_map = {}
    for source in face_sources:
        if not isinstance(source, dict):
            raise RuntimeError("missing-face-sources")

        source_id = source.get("id")
        if source_id is None:
            raise RuntimeError("missing-face-sources")
        source_id = str(source_id)

        source_path = source.get("path")
        if not source_path:
            source_path = override_map.get(source_id)

        if not source_path:
            raise RuntimeError("missing-face-sources")

        _validate_file(
            source_path,
            ALLOWED_IMAGE_EXTS,
            missing_code="unsupported-image-format",
        )
        source_map[source_id] = source_path

    return source_map


def _resolve_target_face(config, args, override_map):
    if args.target_face:
        target_face = args.target_face
    else:
        target_face = config.get("targetFace")
        if isinstance(target_face, dict):
            target_face = target_face.get("path")

    if not target_face:
        target_face_id = config.get("targetFaceId")
        if target_face_id is not None:
            target_face = override_map.get(str(target_face_id))

    if not target_face:
        raise RuntimeError("missing-params")

    _validate_file(
        target_face,
        ALLOWED_IMAGE_EXTS,
        missing_code="unsupported-image-format",
    )
    return target_face


def _resolve_target_faces(config, override_map):
    target_faces = config.get("targetFaces")
    if not isinstance(target_faces, list) or len(target_faces) == 0:
        raise RuntimeError("missing-params")

    resolved = {}
    for index, target in enumerate(target_faces):
        if not isinstance(target, dict):
            raise RuntimeError("missing-params")

        target_id = str(target.get("id") or f"target-{index + 1}")
        target_path = target.get("path")
        if not target_path:
            target_path = override_map.get(target_id)

        if not target_path:
            raise RuntimeError("missing-params")

        _validate_file(
            target_path,
            ALLOWED_IMAGE_EXTS,
            missing_code="unsupported-image-format",
        )
        resolved[target_id] = target_path

    return resolved


def _resolve_segment_options(config, args):
    try:
        segment_duration_sec = int(
            float(config.get("segmentDurationSec", 12) or 12)
        )
    except (TypeError, ValueError):
        segment_duration_sec = 12
    try:
        segment_overlap_frames = int(
            float(config.get("segmentOverlapFrames", 6) or 6)
        )
    except (TypeError, ValueError):
        segment_overlap_frames = 6

    if args.segment_duration_sec is not None:
        segment_duration_sec = int(args.segment_duration_sec)
    if args.segment_overlap_frames is not None:
        segment_overlap_frames = int(args.segment_overlap_frames)

    segment_duration_sec = max(1, segment_duration_sec)
    segment_overlap_frames = max(0, segment_overlap_frames)
    return segment_duration_sec, segment_overlap_frames


def _apply_gpu_overrides(config, args):
    use_gpu = bool(config.get("useGpu", False))
    gpu_provider = _normalize_gpu_provider(str(config.get("gpuProvider", "auto")))

    if args.use_gpu:
        use_gpu = True
    if args.no_gpu:
        use_gpu = False

    if args.gpu_provider:
        gpu_provider = _normalize_gpu_provider(args.gpu_provider)

    if gpu_provider == "cpu":
        use_gpu = False
    elif gpu_provider in ("directml", "cuda"):
        use_gpu = True

    return use_gpu, gpu_provider


def _resolve_key_frame_ms(config, args):
    if args.key_frame_ms is not None:
        return max(0, int(args.key_frame_ms))
    try:
        return max(0, int(float(config.get("keyFrameMs", 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _print_json(payload: dict):
    print(json.dumps(payload, ensure_ascii=False))


def _stage_callback(stage: str):
    print(f"[STAGE] {stage}", flush=True)


def _progress_callback(frame_count: int, total_frames: int, elapsed_seconds: float):
    if total_frames and total_frames > 0:
        progress = max(0.0, min(100.0, frame_count / total_frames * 100.0))
        eta = None
        if frame_count > 0 and elapsed_seconds > 0:
            speed = frame_count / elapsed_seconds
            if speed > 0:
                eta = max(0, int((total_frames - frame_count) / speed))
        print(
            f"[PROGRESS] {frame_count}/{total_frames} ({progress:.2f}%), eta={eta}s",
            flush=True,
        )
        return

    print(
        f"[PROGRESS] frame={frame_count}, elapsed={elapsed_seconds:.1f}s",
        flush=True,
    )


def _maybe_move_output(actual_output: str, requested_output: str | None) -> str:
    if not requested_output:
        return actual_output

    dst = os.path.abspath(requested_output)
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    if os.path.abspath(actual_output) != dst:
        shutil.move(actual_output, dst)
    return dst


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run MagicMirror video face-swap from signed configId in terminal-only mode."
    )
    parser.add_argument(
        "--config-id",
        required=True,
        help="Signed task config token (cfg2.* or legacy cfg1.*)",
    )
    parser.add_argument("--input-video", required=True, help="Input video path on current machine")
    parser.add_argument("--output", help="Optional output path. Defaults to *_output.mp4")
    parser.add_argument("--target-face", help="Override single-face target image path")
    parser.add_argument(
        "--face-source",
        action="append",
        help="Override face source mapping, format: sourceId=/path/to/image",
    )
    parser.add_argument(
        "--library-map-json",
        help="JSON file for id->path mapping, for targetFaceId/faceSources/targetFaces fallback",
    )
    parser.add_argument(
        "--target-face-item",
        action="append",
        help="Override deep target mapping, format: targetId=/path/to/image",
    )
    parser.add_argument(
        "--segment-duration-sec",
        type=int,
        help="Override deep mode segment duration in seconds",
    )
    parser.add_argument(
        "--segment-overlap-frames",
        type=int,
        help="Override deep mode segment overlap frames",
    )
    parser.add_argument("--key-frame-ms", type=int, help="Override key frame ms (multi-face mode)")
    parser.add_argument("--use-gpu", action="store_true", help="Force GPU mode")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--gpu-provider",
        choices=["auto", "cpu", "cuda", "directml", "dml"],
        help="Override GPU provider",
    )
    parser.add_argument(
        "--skip-load-models",
        action="store_true",
        help="Skip explicit model preload (not recommended)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.use_gpu and args.no_gpu:
        _print_json({"status": "failed", "error": "missing-params", "detail": "gpu-conflict"})
        return 2

    try:
        _validate_file(
            args.input_video,
            ALLOWED_VIDEO_EXTS,
            missing_code="unsupported-video-format",
        )

        config = _parse_video_task_config_token(args.config_id)
        if not isinstance(config, dict):
            _print_json({"status": "failed", "error": "config-not-found"})
            return 2

        source_override_map = _build_face_source_override_map(args)
        target_override_map = _build_target_face_override_map(args)
        regions = config.get("regions") if isinstance(config.get("regions"), list) else None
        key_frame_ms = _resolve_key_frame_ms(config, args)
        use_gpu, gpu_provider = _apply_gpu_overrides(config, args)

        expected_input_sha256 = get_expected_input_video_sha256(config)
        if expected_input_sha256 and not verify_file_sha256(args.input_video, expected_input_sha256):
            raise RuntimeError("config-mismatch")

        if not args.skip_load_models:
            if not load_models():
                raise RuntimeError("model-load-failed")

        face_sources = config.get("faceSources")
        target_faces = config.get("targetFaces")

        if isinstance(face_sources, list):
            source_map = _resolve_face_source_map(face_sources, source_override_map)
            expected_source_sha256_map = get_expected_face_source_sha256_map(config)
            for source_id, expected_sha256 in expected_source_sha256_map.items():
                source_path = source_map.get(str(source_id))
                if not source_path or not verify_file_sha256(source_path, expected_sha256):
                    raise RuntimeError("config-mismatch")

            output_path = swap_face_video_by_sources(
                args.input_video,
                source_map,
                regions or [],
                key_frame_ms=key_frame_ms,
                progress_callback=_progress_callback,
                stage_callback=_stage_callback,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
            )
        elif isinstance(target_faces, list):
            target_face_map = _resolve_target_faces(config, target_override_map)
            expected_target_faces_sha256_map = get_expected_target_faces_sha256_map(config)
            for target_id, expected_sha256 in expected_target_faces_sha256_map.items():
                target_path = target_face_map.get(str(target_id))
                if not target_path or not verify_file_sha256(target_path, expected_sha256):
                    raise RuntimeError("config-mismatch")

            segment_duration_sec, segment_overlap_frames = _resolve_segment_options(config, args)
            output_path = swap_face_video_deep(
                args.input_video,
                list(target_face_map.values()),
                regions=regions,
                key_frame_ms=key_frame_ms,
                progress_callback=_progress_callback,
                stage_callback=_stage_callback,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
                segment_duration_sec=segment_duration_sec,
                segment_overlap_frames=segment_overlap_frames,
            )
        else:
            target_face = _resolve_target_face(config, args, target_override_map)
            expected_target_sha256 = get_expected_target_face_sha256(config)
            if expected_target_sha256 and not verify_file_sha256(target_face, expected_target_sha256):
                raise RuntimeError("config-mismatch")

            output_path = swap_face_video(
                args.input_video,
                target_face,
                regions=regions,
                progress_callback=_progress_callback,
                stage_callback=_stage_callback,
                use_gpu=use_gpu,
                gpu_provider=gpu_provider,
            )

        final_output = _maybe_move_output(output_path, args.output)
        _print_json(
            {
                "status": "success",
                "output": final_output,
                "gpu": {"useGpu": use_gpu, "gpuProvider": gpu_provider},
            }
        )
        return 0

    except Exception as e:
        _print_json({"status": "failed", "error": _simplify_task_error(e), "detail": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
