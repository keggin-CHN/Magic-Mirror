# Installation Guide

## Requirements

- Node.js 18+
- Python 3.10+
- Rust and Tauri prerequisites for desktop builds
- Android Studio for Android builds

## Frontend

```bash
npm install
npm run dev
```

## Python Server

```bash
cd src-python
pip install -r requirements.txt
python web_server.py
```

## Desktop App

```bash
npm run tauri dev
```

## Android App

Open the `android-app` directory in Android Studio, wait for Gradle sync, then run the app.

## Download Notes

The in-app server download URL now points to:

https://github.com/keggin-CHN/Magic-Mirror

Model download URLs remain unchanged.
