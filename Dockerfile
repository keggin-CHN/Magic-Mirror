FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src-python/requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download model files used by src-python/magic/face.py (with retries).
ARG SKIP_MODEL_DOWNLOAD=0
RUN mkdir -p models && \
    if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ]; then \
      BASE_URL="https://github.com/idootop/TinyFace/releases/download/models-1.0.0" && \
      for model in arcface_w600k_r50.onnx gfpgan_1.4.onnx inswapper_128_fp16.onnx scrfd_2.5g.onnx; do \
        wget -q --tries=3 --timeout=60 -O "models/${model}" "${BASE_URL}/${model}"; \
      done; \
    fi

COPY src-python/ .

EXPOSE 8023

ENV MIRROR_HOST=0.0.0.0
ENV MIRROR_PORT=8023
ENV WEB_DATA_DIR=/app/data/web
ENV WEB_DIST_DIR=/app/dist-web

CMD ["python", "server.py"]
