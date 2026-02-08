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
  --nofollow-import-to=onnxruntime.capi.onnxruntime_pybind11_state \
  --include-data-files="src-python/models/*.onnx=models/" \
  --output-dir=out src-python/server.py

cd out/server.dist && zip -r ../server.zip .

echo "✅ Done"