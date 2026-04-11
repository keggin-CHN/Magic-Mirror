#!/usr/bin/env python3
import argparse
import base64
import hashlib
import hmac
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON_DIR = PROJECT_ROOT / "src-python"
if str(SRC_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON_DIR))

from magic.face import load_models, swap_face_video, swap_face_video_by_sources  # noqa: E402

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

VIDEO_TASK_CONFIG_TOKEN_PREFIX = "cfg1"
VIDEO_TASK_CONFIG_TTL_SECONDS = int(
    os.environ.get("VIDEO_TASK_CONFIG_TTL_SECONDS", str(7 * 24 * 3600))
)
VIDEO_TASK_CONFIG_SECRET = os.environ.get(
    "VIDEO_TASK_CONFIG_SECRET", "magic-mirror-config-secret"
)


def _clone_json_payload(payload):
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + padding).encode("ascii"))


def _sign_video_task_config_payload(payload_b64: str) -> str:
    return hmac.new(
        VIDEO_TASK_CONFIG_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _parse_video_task_config_token(config_id: str):
    if not isinstance(config_id, str):
        return None
    prefix = f"{VIDEO_TASK_CONFIG_TOKEN_PREFIX}."
    if not config_id.startswith(prefix):
        return None

    parts = config_id.split(".", 2)
    if len(parts) != 3:
        return None

    _, payload_b64, signature = parts
    expected_signature = _sign_video_task_config_payload(payload_b64)
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        raw = _b64url_decode(payload_b64)
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    issued_at = body.get("iat")
    try:
        issued_at = float(issued_at)
    except (TypeError, ValueError):
        return None

    if issued_at <= 0:
        return None
    if time.time() - issued_at > VIDEO_TASK_CONFIG_TTL_SECONDS:
        return None

    config = body.get("config")
    if not isinstance(config, dict):
        return None
    return _clone_json_payload(config)


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
    parser.add_argument("--config-id", required=True, help="Signed task config token (cfg1.*)")
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
        help="JSON file for id->path mapping, for targetFaceId/faceSources fallback",
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

        override_map = _build_face_source_override_map(args)
        regions = config.get("regions") if isinstance(config.get("regions"), list) else None
        key_frame_ms = _resolve_key_frame_ms(config, args)
        use_gpu, gpu_provider = _apply_gpu_overrides(config, args)

        if not args.skip_load_models:
            if not load_models():
                raise RuntimeError("model-load-failed")

        face_sources = config.get("faceSources")
        if isinstance(face_sources, list):
            source_map = _resolve_face_source_map(face_sources, override_map)
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
        else:
            target_face = _resolve_target_face(config, args, override_map)
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