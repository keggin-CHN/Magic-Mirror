#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_DIR="${INSTALL_DIR:-$PROJECT_ROOT}"
WEB_PORT="${WEB_PORT:-21859}"
WEB_UI_PORT="${WEB_UI_PORT:-15129}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
SERVICE_NAME="${SERVICE_NAME:-magic-mirror-web}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
SKIP_WEB_BUILD="${SKIP_WEB_BUILD:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
MODEL_BASE_URL="${MODEL_BASE_URL:-https://github.com/idootop/TinyFace/releases/download/models-1.0.0}"
MODEL_FILES=("arcface_w600k_r50.onnx" "gfpgan_1.4.onnx" "inswapper_128_fp16.onnx" "scrfd_2.5g.onnx")

export DEBIAN_FRONTEND=noninteractive

echo "==> Installing system packages..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  python3 \
  python3-venv \
  python3-pip \
  ffmpeg \
  build-essential \
  nginx \
  rsync

if [ "$INSTALL_DIR" != "$PROJECT_ROOT" ]; then
  echo "==> Copying project to ${INSTALL_DIR}..."
  mkdir -p "$INSTALL_DIR"
  rsync -a --delete \
    --exclude node_modules \
    --exclude .git \
    --exclude dist \
    --exclude dist-web \
    --exclude data \
    "$PROJECT_ROOT"/ "$INSTALL_DIR"/
fi

cd "$INSTALL_DIR"

echo "==> Preparing data directories..."
mkdir -p "$INSTALL_DIR/data/web/uploads" "$INSTALL_DIR/data/web/library"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/data"

if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
  echo "==> Downloading model files..."
  MODEL_DIR="$INSTALL_DIR/src-python/models"
  mkdir -p "$MODEL_DIR"
  for model in "${MODEL_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
      echo "==> Downloading $model..."
      curl -fL "${MODEL_BASE_URL}/${model}" -o "$MODEL_DIR/$model"
    else
      echo "==> Model $model exists; skip."
    fi
  done
  chown -R "$SERVICE_USER:$SERVICE_USER" "$MODEL_DIR"
else
  echo "==> Skip model download."
fi

if [ "$SKIP_WEB_BUILD" != "1" ]; then
  if [ ! -f "dist-web/index.html" ]; then
    echo "==> Building web frontend..."
    if ! command -v node >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt-get install -y nodejs
    fi
    if ! command -v pnpm >/dev/null 2>&1; then
      if command -v corepack >/dev/null 2>&1; then
        corepack enable
        corepack prepare pnpm@8.5.1 --activate
      else
        npm install -g pnpm@8.5.1
      fi
    fi
    pnpm install --no-frozen-lockfile
    VITE_OUT_DIR=dist-web pnpm build
  else
    echo "==> dist-web exists; skip frontend build."
  fi
fi

echo "==> Setting up Python venv..."
python3 -m venv "$INSTALL_DIR/.venv-web"
source "$INSTALL_DIR/.venv-web/bin/activate"
pip install --upgrade pip
pip install -r src-python/requirements.txt
pip install -e src-python --no-deps
deactivate

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> Installing systemd service: ${SERVICE_NAME}"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MagicMirror Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv-web/bin/python $INSTALL_DIR/src-python/web_server.py
Restart=on-failure
Environment=WEB_HOST=$WEB_HOST
Environment=WEB_PORT=$WEB_PORT
User=$SERVICE_USER
Group=$SERVICE_USER

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

NGINX_SITE="/etc/nginx/sites-available/${SERVICE_NAME}"
echo "==> Configuring Nginx for web UI on port ${WEB_UI_PORT}..."
cat > "$NGINX_SITE" <<EOF
server {
  listen ${WEB_UI_PORT};
  server_name _;
  root ${INSTALL_DIR}/dist-web;
  index index.html;
  client_max_body_size 2g;

  location /api/ {
    proxy_pass http://127.0.0.1:${WEB_PORT};
    proxy_http_version 1.1;
    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
  }

  location / {
    try_files \$uri /index.html;
  }
}
EOF

ln -sf "$NGINX_SITE" /etc/nginx/sites-enabled/"${SERVICE_NAME}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo "==> Service status:"
systemctl status "$SERVICE_NAME" --no-pager
systemctl status nginx --no-pager

echo "==> Done."
echo "==> API: http://$WEB_HOST:$WEB_PORT"
echo "==> UI : http://$WEB_HOST:$WEB_UI_PORT"