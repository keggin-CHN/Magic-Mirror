#!/usr/bin/env bash
# publish-release.sh — 全自动统一发布脚本（并发安全）
#
# 所有构建 workflow（server / app / web / android）构建完成后调用本脚本，
# 把各自的产物上传到**同一个** release tag。release 不存在时自动创建，
# 存在时只追加/覆盖资产，绝不删除其他 workflow 上传的资产。
#
# 用法: publish-release.sh <tag> <asset> [<asset>...]
#   例: bash scripts/publish-release.sh "$RELEASE_TAG" dist/*.zip
#
# 并发安全说明:
#   - create 竞态: 两个 job 同时发现 release 不存在并创建时，后创建者
#     报 409 被 `|| true` 吞掉，随后照常 upload。
#   - upload 并发: 各 workflow 资产名互不相同，GitHub API 支持并发上传。
set -euo pipefail

TAG="${1:?usage: publish-release.sh <tag> <asset>...}"
shift
REPO="${GITHUB_REPOSITORY:-keggin-CHN/Magic-Mirror}"

# gh CLI needs GH_TOKEN; GitHub Actions injects GITHUB_TOKEN instead.
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "$GH_TOKEN" ]; then
  echo "publish: GH_TOKEN/GITHUB_TOKEN is not set" >&2
  exit 1
fi

BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT
cat > "$BODY_FILE" <<'EOF'
MagicMirror 全平台完整版 — 所有构建产物由 CI 自动发布于此

## 🖥️ Server（内置 ffmpeg，支持 MOV 音轨复用）
- `server_windows_x86_64.zip` — Windows CPU + DirectML，含模型
- `server_windows_x86_64_cuda.zip` — Windows CUDA 加速（模型自动补下载）
- `server_linux_x86_64.zip` — Linux x86_64，含模型
- `server_linux_aarch64.zip` — Linux ARM64，含模型

## 📱 APP / 桌面客户端
- `MagicMirror_*.exe` — Windows
- `MagicMirror_*.AppImage` / `MagicMirror_*.deb` — Linux x86_64 / ARM64

## 🤖 Android
- `MagicMirror_*_android_arm64-v8a.apk`

## 🌐 Web
- `magicmirror_web_*.tar.gz` — Web 版（内置 ffmpeg）

## 🧠 Models
- `models.zip` + `scrfd_2.5g.onnx` / `arcface_w600k_r50.onnx` / `inswapper_128_fp16.onnx` / `gfpgan_1.4.onnx`
EOF

# 1) 确保 release 存在（并发 create 竞态吞错）
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" --repo "$REPO" \
    --title "MagicMirror $TAG（全平台完整版）" \
    --notes-file "$BODY_FILE" --latest 2>/dev/null || true
fi

# 2) 上传资产（同名覆盖，不删除其他 workflow 的资产）
for f in "$@"; do
  if [ -e "$f" ]; then
    echo "publish: uploading $f"
    gh release upload "$TAG" "$f" --repo "$REPO" --clobber
  else
    echo "publish: skip missing $f"
  fi
done
echo "publish: done -> $TAG"
