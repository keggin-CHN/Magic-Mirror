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
      BASE_URL="https://github.com/keggin-CHN/Magic-Mirror/releases/download/models-1.0.0" && \
      for model in arcface_w600k_r50.onnx gfpgan_1.4.onnx inswapper_128_fp16.onnx scrfd_2.5g.onnx; do \
        wget -q --tries=3 --timeout=60 -O "models/${model}" "${BASE_URL}/${model}" || exit 1; \
      done; \
    fi

COPY src-python/ .

# Run as a non-root user; it needs write access to the data dir (WEB_DATA_DIR).
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/web \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8023

ENV MIRROR_HOST=0.0.0.0
ENV MIRROR_PORT=8023
ENV WEB_DATA_DIR=/app/data/web
ENV WEB_DIST_DIR=/app/dist-web

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8023/status || exit 1

CMD ["python", "server.py"]
