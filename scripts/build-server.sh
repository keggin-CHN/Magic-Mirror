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

cd out/server.dist && zip -r ../server.zip .

echo "✅ Done"