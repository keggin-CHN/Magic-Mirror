# Usage Guide

## Basic Workflow

1. Start the frontend and Python server.
2. Upload a clear front-facing photo as the face source.
3. Upload the target image or video.
4. Choose single-face or multi-face mode.
5. Start the face swap and wait for processing to finish.
6. Download the generated result.

## Multi-Face Swap

In multi-face mode, you can add multiple face sources and assign different faces to selected regions.

## Video Face Swap

Video processing can take a long time. GPU acceleration is recommended. Use DirectML on Windows or CUDA for supported NVIDIA GPUs.

## Notes

- Use clear, unobstructed, front-facing photos.
- HEIC/HEIF is not supported. Convert to JPG or PNG first.
- MP4/H.264 videos are recommended.
