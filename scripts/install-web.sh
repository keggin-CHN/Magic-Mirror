#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi

INSTALL_DIR="${INSTALL_DIR:-$PROJECT_ROOT}"
WEB_PORT="${WEB_PORT:-21859}"
WEB_UI_PORT="${WEB_UI_PORT:-15129}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
SERVICE_NAME="${SERVICE_NAME:-magic-mirror-web}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
WEB_SERVER_DIR="${WEB_SERVER_DIR:-$INSTALL_DIR/web_server.dist}"

export DEBIAN_FRONTEND=noninteractive

echo "==> Switching apt source to Aliyun..."
if [ -f /etc/os-release ]; then
  . /etc/os-release
  CODENAME="${VERSION_CODENAME:-}"
  if [ -z "$CODENAME" ] && command -v lsb_release >/dev/null 2>&1; then
    CODENAME="$(lsb_release -cs || true)"
  fi
  if [ -n "$CODENAME" ]; then
    if [ -f /etc/apt/sources.list ]; then
      cp /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)" || true
    fi
    if [ "$ID" = "ubuntu" ]; then
      cat > /etc/apt/sources.list <<EOF
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME} main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-security main restricted universe multiverse
EOF
    elif [ "$ID" = "debian" ]; then
      cat > /etc/apt/sources.list <<EOF
deb http://mirrors.aliyun.com/debian/ ${CODENAME} main contrib non-free non-free-firmware
deb http://mirrors.aliyun.com/debian/ ${CODENAME}-updates main contrib non-free non-free-firmware
deb http://mirrors.aliyun.com/debian-security ${CODENAME}-security main contrib non-free non-free-firmware
EOF
    else
      echo "==> Unknown distro: $ID, skip mirror switch."
    fi
  else
    echo "==> Could not detect codename, skip mirror switch."
  fi
else
  echo "==> /etc/os-release missing, skip mirror switch."
fi

echo "==> Installing system packages..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  ffmpeg \
  nginx \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxrender1 \
  libxext6 \
  rsync \
  unzip \
  procps

# Auto-extract artifact zip or tarball if present
if [ -f "${PROJECT_ROOT}/web_debian_ubuntu_x86_64.zip" ]; then
  echo "==> Found artifact zip, extracting..."
  unzip -o "${PROJECT_ROOT}/web_debian_ubuntu_x86_64.zip" -d "$PROJECT_ROOT"
fi

TARBALL=$(find "$PROJECT_ROOT" -maxdepth 1 -name "magicmirror_web_*_debian_ubuntu_x86_64.tar.gz" | head -n 1)
if [ -n "$TARBALL" ]; then
  if [ ! -d "$PROJECT_ROOT/web_server.dist" ] || [ ! -d "$PROJECT_ROOT/dist-web" ]; then
    echo "==> Extracting tarball $TARBALL..."
    tar -xzf "$TARBALL" -C "$PROJECT_ROOT"
  fi
fi

if [ "$INSTALL_DIR" != "$PROJECT_ROOT" ]; then
  echo "==> Copying project to ${INSTALL_DIR}..."
  mkdir -p "$INSTALL_DIR"
  rsync -a --delete \
    --exclude node_modules \
    --exclude .git \
    --exclude dist \
    --exclude data \
    "$PROJECT_ROOT"/ "$INSTALL_DIR"/
fi

cd "$INSTALL_DIR"

echo "==> Preparing data directories..."
mkdir -p "$INSTALL_DIR/data/web/uploads" "$INSTALL_DIR/data/web/library"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/data"

if [ ! -d "$WEB_SERVER_DIR" ]; then
  echo "web_server.dist not found in $WEB_SERVER_DIR" >&2
  echo "Please extract the official web bundle which contains web_server.dist." >&2
  exit 1
fi

if [ ! -f "$INSTALL_DIR/dist-web/index.html" ]; then
  echo "dist-web not found. Please extract the web bundle first." >&2
  exit 1
fi

WEB_SERVER_BIN="$WEB_SERVER_DIR/web_server.bin"
if [ ! -f "$WEB_SERVER_BIN" ]; then
  WEB_SERVER_BIN="$WEB_SERVER_DIR/web_server"
fi
if [ ! -f "$WEB_SERVER_BIN" ]; then
  echo "web server binary not found in $WEB_SERVER_DIR" >&2
  exit 1
fi
chmod +x "$WEB_SERVER_BIN"

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

if command -v systemctl >/dev/null 2>&1; then
  SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

  echo "==> Installing systemd service: ${SERVICE_NAME}"
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MagicMirror Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$WEB_SERVER_BIN
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
  
  systemctl enable --now nginx
  systemctl reload nginx

  echo "==> Service status:"
  systemctl status "$SERVICE_NAME" --no-pager
  systemctl status nginx --no-pager

  echo "==> Done."
  echo "==> API: http://$WEB_HOST:$WEB_PORT"
  echo "==> UI : http://$WEB_HOST:$WEB_UI_PORT"
else
  echo "==> systemctl not found, skipping systemd service installation."
  echo "==> Starting Nginx manually..."
  if pgrep nginx >/dev/null; then
    nginx -s reload
  else
    nginx
  fi

  echo "==> Done."
  echo "==> API: http://$WEB_HOST:$WEB_PORT"
  echo "==> UI : http://$WEB_HOST:$WEB_UI_PORT"
  
  echo "==> Starting Web Server in foreground..."
  export WEB_HOST
  export WEB_PORT
  exec "$WEB_SERVER_BIN"
fi