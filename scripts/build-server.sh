#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="${OUT_DIR:-out}"
DIST_DIR="${OUT_DIR}/server.dist"
ARCHIVE_PATH="${OUT_DIR}/server.zip"

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

log "Building server..."

PYTHONOPTIMIZE=1 python -O -m nuitka --standalone --unstripped --assume-yes-for-downloads \
  --include-package=onnx \
  --include-package=google.protobuf \
  --include-package=onnxruntime \
  --include-package-data=onnxruntime \
  --include-package=async_tasks \
  --include-package=cv2 \
  --include-package=numpy \
  --include-package=tinyface \
  --include-package=fastapi \
  --include-package=uvicorn \
  --include-package=multipart \
  --include-package=av \
  --include-package-data=onnx \
  --include-data-files="src-python/models/*.onnx=models/" \
  --output-dir="$OUT_DIR" \
  src-python/server.py

[ -d "$DIST_DIR" ] || die "Nuitka output directory not found: $DIST_DIR"

log "Copying GPU diagnostic scripts..."
for file in check_gpu_support.bat check_gpu_support.py; do
  if [ -f "$file" ]; then
    cp "$file" "$DIST_DIR/"
  else
    echo "[WARN] Optional file not found: $file" >&2
  fi
done

log "Bundling static ffmpeg (for audio track muxing)..."
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" ;;
  aarch64 | arm64) FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz" ;;
  *)
    echo "[ERROR] Unsupported architecture for static ffmpeg: $ARCH" >&2
    exit 1
    ;;
esac
if [ ! -x "$DIST_DIR/ffmpeg" ]; then
  FFMPEG_TARBALL="${OUT_DIR}/ffmpeg-static.tar.xz"
  curl -fL --retry 3 --retry-delay 10 -o "$FFMPEG_TARBALL" "$FFMPEG_URL"
  # Guard against CDN/WAF intercept pages being saved instead of the archive.
  file "$FFMPEG_TARBALL" | grep -q "XZ" || die "ffmpeg download failed: not an XZ archive ($(file -b "$FFMPEG_TARBALL"))"
  tar -xJf "$FFMPEG_TARBALL" -C "$OUT_DIR"
  # BtbN archives extract to a fixed directory name; avoid `find` here so a
  # permission error on an unrelated dir cannot fail the build (bash -e).
  case "$ARCH" in
    x86_64) FFMPEG_DIR="${OUT_DIR}/ffmpeg-master-latest-linux64-gpl" ;;
    *) FFMPEG_DIR="${OUT_DIR}/ffmpeg-master-latest-linuxarm64-gpl" ;;
  esac
  [ -x "$FFMPEG_DIR/bin/ffmpeg" ] || die "ffmpeg binary not found in downloaded archive"
  cp "$FFMPEG_DIR/bin/ffmpeg" "$DIST_DIR/ffmpeg"
  chmod +x "$DIST_DIR/ffmpeg"
  if [ -x "$FFMPEG_DIR/bin/ffprobe" ]; then
    cp "$FFMPEG_DIR/bin/ffprobe" "$DIST_DIR/ffprobe"
    chmod +x "$DIST_DIR/ffprobe"
  fi
  rm -f "$FFMPEG_TARBALL"
  rm -rf "$FFMPEG_DIR"
  log "Bundled ffmpeg for $ARCH"
else
  log "ffmpeg already present in $DIST_DIR"
fi

log "Packaging server archive..."
rm -f "$ARCHIVE_PATH"
(
  cd "$DIST_DIR"
  zip -r "../server.zip" .
)

[ -f "$ARCHIVE_PATH" ] || die "Package was not created: $ARCHIVE_PATH"

log "Created $ARCHIVE_PATH"
