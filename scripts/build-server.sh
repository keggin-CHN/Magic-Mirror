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

log "Packaging server archive..."
rm -f "$ARCHIVE_PATH"
(
  cd "$DIST_DIR"
  zip -r "../server.zip" .
)

[ -f "$ARCHIVE_PATH" ] || die "Package was not created: $ARCHIVE_PATH"

log "Created $ARCHIVE_PATH"
