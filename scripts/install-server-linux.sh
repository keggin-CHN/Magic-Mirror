#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "[ERROR] Please run as root (sudo)." >&2
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
TERMINAL_ONLY_MODE="${TERMINAL_ONLY_MODE:-0}"
WEB_HOST="${WEB_HOST:-}"
SERVICE_NAME="${SERVICE_NAME:-magic-mirror-web}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
WEB_SERVER_DIR="${WEB_SERVER_DIR:-$INSTALL_DIR/web_server.dist}"
USE_ALIYUN_MIRROR="${USE_ALIYUN_MIRROR:-0}"
SKIP_NGINX="${SKIP_NGINX:-}"
VIDEO_TASK_CONFIG_SECRET="${VIDEO_TASK_CONFIG_SECRET:-magic-mirror-config-secret}"

if [ "$TERMINAL_ONLY_MODE" = "1" ]; then
  WEB_HOST="${WEB_HOST:-127.0.0.1}"
  SKIP_NGINX="${SKIP_NGINX:-1}"
else
  WEB_HOST="${WEB_HOST:-0.0.0.0}"
  SKIP_NGINX="${SKIP_NGINX:-0}"
fi

PKG_MANAGER=""

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

detect_pkg_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt"
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    return
  fi
  if command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    return
  fi
  err "Unsupported Linux distro: no apt-get/dnf/yum found."
  exit 1
}

ensure_service_user() {
  if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    warn "Service user '$SERVICE_USER' does not exist. Fallback to root."
    SERVICE_USER="root"
  fi
}

configure_apt_mirror_if_needed() {
  if [ "$PKG_MANAGER" != "apt" ] || [ "$USE_ALIYUN_MIRROR" != "1" ]; then
    return
  fi
  log "USE_ALIYUN_MIRROR=1, trying to switch apt mirror to Aliyun."

  if [ ! -f /etc/os-release ]; then
    warn "/etc/os-release not found, skip apt mirror switch."
    return
  fi

  # shellcheck disable=SC1091
  . /etc/os-release
  CODENAME="${VERSION_CODENAME:-}"
  if [ -z "$CODENAME" ] && command -v lsb_release >/dev/null 2>&1; then
    CODENAME="$(lsb_release -cs || true)"
  fi

  if [ -z "$CODENAME" ]; then
    warn "Cannot detect codename, skip apt mirror switch."
    return
  fi

  if [ -f /etc/apt/sources.list ]; then
    cp /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)" || true
  fi

  if [ "${ID:-}" = "ubuntu" ]; then
    cat > /etc/apt/sources.list <<EOF
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME} main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-security main restricted universe multiverse
EOF
  elif [ "${ID:-}" = "debian" ]; then
    cat > /etc/apt/sources.list <<EOF
deb http://mirrors.aliyun.com/debian/ ${CODENAME} main contrib non-free non-free-firmware
deb http://mirrors.aliyun.com/debian/ ${CODENAME}-updates main contrib non-free non-free-firmware
deb http://mirrors.aliyun.com/debian-security ${CODENAME}-security main contrib non-free non-free-firmware
EOF
  else
    warn "Distro '${ID:-unknown}' is not ubuntu/debian, skip apt mirror switch."
  fi
}

install_packages_apt() {
  export DEBIAN_FRONTEND=noninteractive
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
}

install_packages_dnf() {
  dnf -y makecache || true
  dnf -y install epel-release || true

  dnf -y install \
    ca-certificates \
    curl \
    git \
    nginx \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    glib2 \
    libgomp \
    libSM \
    libXrender \
    libXext \
    rsync \
    unzip \
    procps-ng || true

  if ! dnf -y install ffmpeg; then
    warn "dnf install ffmpeg failed. Install ffmpeg manually if audio mux is required."
  fi
}

install_packages_yum() {
  yum -y makecache fast || true
  yum -y install epel-release || true

  yum -y install \
    ca-certificates \
    curl \
    git \
    nginx \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    glib2 \
    libgomp \
    libSM \
    libXrender \
    libXext \
    rsync \
    unzip \
    procps-ng || true

  if ! yum -y install ffmpeg; then
    warn "yum install ffmpeg failed. Install ffmpeg manually if audio mux is required."
  fi
}

install_system_packages() {
  log "Installing system packages via ${PKG_MANAGER} ..."
  case "$PKG_MANAGER" in
    apt) install_packages_apt ;;
    dnf) install_packages_dnf ;;
    yum) install_packages_yum ;;
    *)
      err "Unknown package manager: $PKG_MANAGER"
      exit 1
      ;;
  esac
}

extract_bundle_if_needed() {
  if [ -f "${PROJECT_ROOT}/web_debian_ubuntu_x86_64.zip" ]; then
    log "Found legacy bundle zip: web_debian_ubuntu_x86_64.zip"
    unzip -o "${PROJECT_ROOT}/web_debian_ubuntu_x86_64.zip" -d "$PROJECT_ROOT"
  fi

  if [ -f "${PROJECT_ROOT}/web_linux_x86_64.zip" ]; then
    log "Found bundle zip: web_linux_x86_64.zip"
    unzip -o "${PROJECT_ROOT}/web_linux_x86_64.zip" -d "$PROJECT_ROOT"
  fi

  TARBALL="$(find "$PROJECT_ROOT" -maxdepth 1 -name "magicmirror_web_*.tar.gz" | head -n 1 || true)"
  if [ -n "$TARBALL" ]; then
    if [ ! -d "$PROJECT_ROOT/web_server.dist" ] || [ ! -d "$PROJECT_ROOT/dist-web" ]; then
      log "Extracting tarball: $TARBALL"
      tar -xzf "$TARBALL" -C "$PROJECT_ROOT"
    fi
  fi
}

sync_install_dir() {
  if [ "$INSTALL_DIR" = "$PROJECT_ROOT" ]; then
    return
  fi
  log "Sync project to INSTALL_DIR=$INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"
  rsync -a --delete \
    --exclude node_modules \
    --exclude .git \
    --exclude dist \
    --exclude data \
    "$PROJECT_ROOT"/ "$INSTALL_DIR"/
}

ensure_runtime_layout() {
  mkdir -p "$INSTALL_DIR/data/web/uploads" "$INSTALL_DIR/data/web/library"
  chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/data" || true
}

resolve_web_server_bin() {
  WEB_SERVER_BIN="$WEB_SERVER_DIR/web_server.bin"
  if [ ! -f "$WEB_SERVER_BIN" ]; then
    WEB_SERVER_BIN="$WEB_SERVER_DIR/web_server"
  fi
  if [ ! -f "$WEB_SERVER_BIN" ]; then
    err "web server binary not found in $WEB_SERVER_DIR"
    exit 1
  fi
  chmod +x "$WEB_SERVER_BIN"
}

configure_nginx_site() {
  if [ "$SKIP_NGINX" = "1" ]; then
    warn "SKIP_NGINX=1, skip nginx setup."
    return
  fi

  if ! command -v nginx >/dev/null 2>&1; then
    err "nginx command not found. Install nginx or set SKIP_NGINX=1."
    exit 1
  fi

  NGINX_CONF=""
  if [ -d /etc/nginx/sites-available ] && [ -d /etc/nginx/sites-enabled ]; then
    NGINX_CONF="/etc/nginx/sites-available/${SERVICE_NAME}"
    cat > "$NGINX_CONF" <<EOF
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
    ln -sf "$NGINX_CONF" "/etc/nginx/sites-enabled/${SERVICE_NAME}"
    rm -f /etc/nginx/sites-enabled/default || true
  else
    mkdir -p /etc/nginx/conf.d
    NGINX_CONF="/etc/nginx/conf.d/${SERVICE_NAME}.conf"
    cat > "$NGINX_CONF" <<EOF
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
  fi

  nginx -t
}

install_or_start_services() {
  if command -v systemctl >/dev/null 2>&1; then
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    log "Installing systemd service: ${SERVICE_NAME}"

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MagicMirror Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=${WEB_SERVER_BIN}
Restart=on-failure
Environment=WEB_HOST=${WEB_HOST}
Environment=WEB_PORT=${WEB_PORT}
Environment=VIDEO_TASK_CONFIG_SECRET=${VIDEO_TASK_CONFIG_SECRET}
User=${SERVICE_USER}
Group=${SERVICE_USER}

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable --now "$SERVICE_NAME"

    if [ "$SKIP_NGINX" != "1" ]; then
      systemctl enable --now nginx
      systemctl reload nginx
    fi
    return
  fi

  warn "systemctl not found. Use fallback process start."

  if [ "$SKIP_NGINX" != "1" ]; then
    if pgrep nginx >/dev/null 2>&1; then
      nginx -s reload
    else
      nginx
    fi
  fi

  LOG_DIR="$INSTALL_DIR/data/web"
  mkdir -p "$LOG_DIR"
  PID_FILE="$LOG_DIR/${SERVICE_NAME}.pid"
  LOG_FILE="$LOG_DIR/${SERVICE_NAME}.log"

  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    warn "Existing process detected with PID $(cat "$PID_FILE"), skip restart."
    return
  fi

  log "Starting web server by nohup (no systemd)."
  nohup env \
    WEB_HOST="$WEB_HOST" \
    WEB_PORT="$WEB_PORT" \
    VIDEO_TASK_CONFIG_SECRET="$VIDEO_TASK_CONFIG_SECRET" \
    "$WEB_SERVER_BIN" >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
}

print_summary() {
  log "Done."
  if [ "$TERMINAL_ONLY_MODE" = "1" ]; then
    log "Terminal-only mode enabled (no public web entry by default)."
  fi
  log "API endpoint: http://${WEB_HOST}:${WEB_PORT}"
  if [ "$SKIP_NGINX" != "1" ]; then
    log "Web UI endpoint: http://${WEB_HOST}:${WEB_UI_PORT}"
  else
    log "Nginx disabled."
  fi
  log "Service name: ${SERVICE_NAME}"
  log "Install dir: ${INSTALL_DIR}"
}

main() {
  detect_pkg_manager
  ensure_service_user
  configure_apt_mirror_if_needed
  install_system_packages
  extract_bundle_if_needed
  sync_install_dir

  cd "$INSTALL_DIR"

  if [ ! -d "$WEB_SERVER_DIR" ]; then
    err "web_server.dist not found in $WEB_SERVER_DIR"
    err "Please extract official web bundle first."
    exit 1
  fi
  if [ "$SKIP_NGINX" != "1" ]; then
    if [ ! -f "$INSTALL_DIR/dist-web/index.html" ]; then
      err "dist-web not found in $INSTALL_DIR/dist-web"
      err "Please extract official web bundle first."
      exit 1
    fi
  else
    if [ ! -f "$INSTALL_DIR/dist-web/index.html" ]; then
      warn "dist-web not found, but SKIP_NGINX=1 so UI bundle is optional."
    fi
  fi

  ensure_runtime_layout
  resolve_web_server_bin
  configure_nginx_site
  install_or_start_services
  print_summary
}

main "$@"