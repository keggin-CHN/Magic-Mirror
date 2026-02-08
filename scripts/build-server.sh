#!/bin/bash
set -e  # 遇到错误立即退出

echo "🔨 开始构建服务器..."

python -m nuitka --standalone --assume-yes-for-downloads \
  --include-package=onnx \
  --include-package=google.protobuf \
  --include-package=onnxruntime \
  --include-package-data=onnxruntime \
  --include-package=async_tasks \
  --include-package=cv2 \
  --include-package=numpy \
  --include-package=tinyface \
  --include-package=bottle \
  --include-package-data=onnx \
  --include-data-files="src-python/models/*.onnx=models/" \
  --output-dir=out src-python/server.py

if [ ! -d "out/server.dist" ]; then
    echo "❌ 构建失败：out/server.dist 目录不存在"
    exit 1
fi

echo "📦 打包服务器..."
cd out/server.dist && zip -r ../server.zip .

if [ ! -f "../server.zip" ]; then
    echo "❌ 打包失败：server.zip 文件不存在"
    exit 1
fi

cd ../..

# 复制 GPU 检测脚本到输出目录
echo "📋 复制 GPU 检测脚本..."
if [ ! -f "check_gpu_support.bat" ]; then
    echo "⚠️  警告：check_gpu_support.bat 不存在"
else
    cp check_gpu_support.bat out/server.dist/
fi

if [ ! -f "check_gpu_support.py" ]; then
    echo "⚠️  警告：check_gpu_support.py 不存在"
else
    cp check_gpu_support.py out/server.dist/
fi

echo "✅ Done"