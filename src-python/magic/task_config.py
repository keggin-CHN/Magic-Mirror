import base64
import hashlib
import hmac
import json
import os
from typing import Any, Dict, Iterable, List, Optional

VIDEO_TASK_CONFIG_TOKEN_PREFIX = "cfg2"
LEGACY_VIDEO_TASK_CONFIG_TOKEN_PREFIX = "cfg1"


def clone_json_payload(payload: Any):
    """Deep clone a JSON-serializable payload."""
    return json.loads(json.dumps(payload, ensure_ascii=False))


def b64url_encode(raw: bytes) -> str:
    """Encode bytes to URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64url_decode(encoded: str) -> bytes:
    """Decode URL-safe base64 without padding."""
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + padding).encode("ascii"))


def sign_video_task_config_payload(payload_b64: str, secret: str) -> str:
    """Sign a payload with HMAC-SHA256."""
    return hmac.new(
        str(secret).encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def compute_file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_sha256(path: str, expected_sha256: Optional[str]) -> bool:
    """Verify a file matches its expected SHA256 hash."""
    normalized = _normalize_sha256(expected_sha256)
    if not normalized:
        return True
    return compute_file_sha256(path) == normalized


def canonicalize_video_task_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a video task config to a canonical form."""
    if not isinstance(payload, dict):
        raise RuntimeError("missing-params")

    input_video = _canonicalize_input_video(payload)
    if input_video is None:
        raise RuntimeError("missing-params")

    face_sources = payload.get("faceSources")
    has_face_sources = isinstance(face_sources, list) and len(face_sources) > 0
    target_faces = payload.get("targetFaces")
    has_target_faces = isinstance(target_faces, list) and len(target_faces) > 0

    regions = _canonicalize_regions(payload.get("regions"))
    key_frame_ms = _coerce_non_negative_int(payload.get("keyFrameMs"))
    use_gpu = bool(payload.get("useGpu", False))
    gpu_provider = _normalize_gpu_provider(payload.get("gpuProvider"))
    deep_swap_mode = bool(payload.get("deepSwapMode", False) or has_target_faces)
    segment_duration_sec = _coerce_non_negative_int(payload.get("segmentDurationSec"))
    segment_overlap_frames = _coerce_non_negative_int(payload.get("segmentOverlapFrames"))

    out: Dict[str, Any] = {
        "schema": "magicmirror.video-task",
        "version": 2,
        "inputVideo": input_video,
        "regions": regions,
        "keyFrameMs": key_frame_ms,
        "useGpu": use_gpu,
        "gpuProvider": gpu_provider,
    }

    if has_face_sources:
        out["swapMode"] = "multi-source"
        out["faceSources"] = _canonicalize_face_sources(face_sources)
    elif has_target_faces:
        out["swapMode"] = "deep"
        out["deepSwapMode"] = deep_swap_mode
        out["targetFaces"] = _canonicalize_face_sources(target_faces)
        out["segmentDurationSec"] = segment_duration_sec
        out["segmentOverlapFrames"] = segment_overlap_frames
    else:
        target_face = _canonicalize_target_face(payload)
        if target_face is None:
            raise RuntimeError("missing-params")
        out["swapMode"] = "single"
        out["targetFace"] = target_face

    return out


def build_video_task_config_token(payload: Dict[str, Any], secret: str) -> str:
    """Build a signed config token for a video task."""
    canonical = canonicalize_video_task_config(payload)
    raw = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload_b64 = b64url_encode(raw)
    signature = sign_video_task_config_payload(payload_b64, secret)
    return f"{VIDEO_TASK_CONFIG_TOKEN_PREFIX}.{payload_b64}.{signature}"


def parse_video_task_config_token(
    config_id: str,
    secret: str,
    *,
    legacy_ttl_seconds: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(config_id, str):
        return None

    parsed = _parse_signed_token(
        config_id,
        prefix=VIDEO_TASK_CONFIG_TOKEN_PREFIX,
        secret=secret,
    )
    if parsed is not None:
        if isinstance(parsed, dict) and parsed.get("schema") == "magicmirror.video-task":
            return clone_json_payload(parsed)
        return None

    legacy = _parse_signed_token(
        config_id,
        prefix=LEGACY_VIDEO_TASK_CONFIG_TOKEN_PREFIX,
        secret=secret,
    )
    if not isinstance(legacy, dict):
        return None

    issued_at = legacy.get("iat")
    if legacy_ttl_seconds is not None:
        try:
            issued_at = float(issued_at)
        except (TypeError, ValueError):
            return None
        if issued_at <= 0:
            return None
        import time

        if time.time() - issued_at > float(legacy_ttl_seconds):
            return None

    config = legacy.get("config")
    if not isinstance(config, dict):
        return None
    return clone_json_payload(config)


def get_expected_input_video_sha256(config: Dict[str, Any]) -> Optional[str]:
    """Get the expected SHA256 for the input video."""
    return _extract_sha256(config.get("inputVideo"))


def get_expected_target_face_sha256(config: Dict[str, Any]) -> Optional[str]:
    """Get the expected SHA256 for the target face."""
    return _extract_sha256(config.get("targetFace"))


def get_expected_target_faces_sha256_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Get a map of target face SHA256 hashes."""
    out: Dict[str, str] = {}
    target_faces = config.get("targetFaces")
    if not isinstance(target_faces, list):
        return out

    for item in target_faces:
        if not isinstance(item, dict):
            continue
        source_id = item.get("id")
        sha256 = _extract_sha256(item)
        if source_id is None or not sha256:
            continue
        out[str(source_id)] = sha256
    return out


def get_expected_face_source_sha256_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Get a map of face source SHA256 hashes."""
    out: Dict[str, str] = {}
    face_sources = config.get("faceSources")
    if not isinstance(face_sources, list):
        return out

    for item in face_sources:
        if not isinstance(item, dict):
            continue
        source_id = item.get("id")
        sha256 = _extract_sha256(item)
        if source_id is None or not sha256:
            continue
        out[str(source_id)] = sha256
    return out


def _parse_signed_token(config_id: str, *, prefix: str, secret: str) -> Optional[Dict[str, Any]]:
    """Parse and verify a signed token."""
    token_prefix = f"{prefix}."
    if not config_id.startswith(token_prefix):
        return None

    parts = config_id.split(".", 2)
    if len(parts) != 3:
        return None

    _, payload_b64, signature = parts
    expected_signature = sign_video_task_config_payload(payload_b64, secret)
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        raw = b64url_decode(payload_b64)
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(body, dict):
        return None
    return body


def _canonicalize_input_video(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Canonicalize input video config."""
    input_sha256 = _normalize_sha256(payload.get("inputVideoHash"))
    if input_sha256:
        return {"sha256": input_sha256}

    input_video = payload.get("inputVideo")
    if isinstance(input_video, dict):
        sha256 = _extract_sha256(input_video)
        if sha256:
            return {"sha256": sha256}
        input_video = input_video.get("path")

    if isinstance(input_video, str) and input_video:
        if not os.path.exists(input_video):
            raise FileNotFoundError("file-not-found")
        return {"sha256": compute_file_sha256(input_video)}

    return None


def _canonicalize_target_face(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Canonicalize target face config."""
    target_sha256 = _normalize_sha256(payload.get("targetFaceHash"))
    if target_sha256:
        return {"sha256": target_sha256}

    target_face = payload.get("targetFace")
    if isinstance(target_face, dict):
        sha256 = _extract_sha256(target_face)
        if sha256:
            return {"sha256": sha256}
        target_face = target_face.get("path")

    if isinstance(target_face, str) and target_face:
        if not os.path.exists(target_face):
            raise FileNotFoundError("file-not-found")
        return {"sha256": compute_file_sha256(target_face)}

    return None


def _canonicalize_face_sources(face_sources: Iterable[Any]) -> List[Dict[str, str]]:
    """Canonicalize face source configs."""
    normalized: List[Dict[str, str]] = []

    for item in face_sources:
        if not isinstance(item, dict):
            raise RuntimeError("missing-face-sources")

        source_id = item.get("id")
        if source_id is None or str(source_id).strip() == "":
            raise RuntimeError("missing-face-sources")

        sha256 = _extract_sha256(item)
        if not sha256:
            source_path = item.get("path")
            if not source_path:
                raise RuntimeError("missing-face-sources")
            if not os.path.exists(source_path):
                raise FileNotFoundError("file-not-found")
            sha256 = compute_file_sha256(source_path)

        normalized.append(
            {
                "id": str(source_id),
                "sha256": sha256,
            }
        )

    normalized.sort(key=lambda item: item["id"])
    return normalized


def _canonicalize_regions(regions: Any) -> List[Dict[str, Any]]:
    """Canonicalize face regions."""
    if not isinstance(regions, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in regions:
        if not isinstance(item, dict):
            continue

        width = _coerce_non_negative_int(item.get("width"))
        height = _coerce_non_negative_int(item.get("height"))
        if width <= 0 or height <= 0:
            continue

        region: Dict[str, Any] = {
            "x": _coerce_non_negative_int(item.get("x")),
            "y": _coerce_non_negative_int(item.get("y")),
            "width": width,
            "height": height,
        }

        face_source_id = item.get("faceSourceId")
        if face_source_id is not None and str(face_source_id).strip() != "":
            region["faceSourceId"] = str(face_source_id)

        normalized.append(region)

    normalized.sort(
        key=lambda item: (
            str(item.get("faceSourceId", "")),
            int(item["x"]),
            int(item["y"]),
            int(item["width"]),
            int(item["height"]),
        )
    )
    return normalized


def _coerce_non_negative_int(value: Any) -> int:
    """Coerce a value to a non-negative int."""
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _normalize_gpu_provider(value: Any) -> str:
    """Normalize GPU provider name."""
    mode = str(value or "auto").strip().lower()
    if mode in {"dml", "directml"}:
        return "directml"
    if mode == "cuda":
        return "cuda"
    if mode == "cpu":
        return "cpu"
    return "auto"


def _extract_sha256(item: Any) -> Optional[str]:
    """Extract SHA256 hash from an item."""
    if not isinstance(item, dict):
        return None

    for key in ("sha256", "hash", "contentHash"):
        value = _normalize_sha256(item.get(key))
        if value:
            return value
    return None


def _normalize_sha256(value: Any) -> Optional[str]:
    """Normalize a SHA256 hash value."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        return None
    if any(ch not in "0123456789abcdef" for ch in normalized):
        return None
    return normalized