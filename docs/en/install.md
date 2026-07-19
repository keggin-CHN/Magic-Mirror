# Installation Guide

## Requirements

- Node.js 18+
- pnpm 8+
- Python 3.10+
- Rust and Tauri prerequisites for desktop builds
- Android Studio for Android builds

## Frontend

```bash
pnpm install
pnpm dev
```

## Python Server

```bash
cd src-python
pip install -r requirements.txt
python web_server.py
```

When running from source, the Web server keeps `123456` as the local development default if no config exists. The production Linux Web installer generates a one-time random initial password instead.

## Linux Web Production Install

Use the installer from the official Web bundle:

```bash
sudo ./install-web.sh
```

The installer generates a random `VIDEO_TASK_CONFIG_SECRET` unless one is provided, creates the initial credential hash when no config exists, and does not store the plaintext initial password in systemd.

## Desktop App

```bash
pnpm tauri dev
```

## Android App

Open the `android-app` directory in Android Studio, wait for Gradle sync, then run the app.

## Download Notes

The in-app server download URL now points to:

https://github.com/keggin-CHN/Magic-Mirror

Model download URLs remain unchanged.
