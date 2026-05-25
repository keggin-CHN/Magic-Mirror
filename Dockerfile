FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src-python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model files
RUN mkdir -p models && \
    wget -q --show-progress -O models/det_500m.onnx \
      "https://github.com/idootop/TinyFace/releases/download/models-1.0.0/det_500m.onnx" && \
    wget -q --show-progress -O models/w600k_r50.onnx \
      "https://github.com/idootop/TinyFace/releases/download/models-1.0.0/w600k_r50.onnx" && \
    wget -q --show-progress -O models/inswapper_128.onnx \
      "https://github.com/idootop/TinyFace/releases/download/models-1.0.0/inswapper_128.onnx"

COPY src-python/ .

EXPOSE 8023

ENV MIRROR_HOST=0.0.0.0
ENV MIRROR_PORT=8023

CMD ["python", "-m", "magic.app"]
