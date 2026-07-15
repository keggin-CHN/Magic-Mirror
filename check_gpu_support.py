#!/usr/bin/env python3
"""Verify ONNX Runtime GPU providers on Windows.

Provider names alone are not enough: a CPU-only ONNX Runtime build, or a
missing CUDA/cuDNN DLL, can still report ``CUDAExecutionProvider`` as available.
This script creates a tiny ONNX model and verifies that the selected provider
can initialize a real inference session.
"""

from __future__ import annotations

import os
import sys
import tempfile


def _configure_console_encoding() -> None:
    """Keep the diagnostic usable under the default Windows GBK console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


_configure_console_encoding()


def _probe_provider(ort, provider: str) -> tuple[bool, str]:
    """Initialize and run a tiny ONNX session with a requested provider."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node("Add", ["x", "y"], ["z"])
        graph = helper.make_graph(
            [node],
            "magic_mirror_gpu_probe",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
            ],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13)],
        )
        model.ir_version = min(model.ir_version, 10)

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
                path = handle.name
            onnx.save(model, path)
            session = ort.InferenceSession(
                path,
                providers=[provider, "CPUExecutionProvider"],
            )
            active = session.get_providers() or []
            if provider not in active:
                return False, f"provider fell back to {active}"
            result = session.run(
                None,
                {
                    "x": np.array([1, 2], dtype=np.float32),
                    "y": np.array([3, 4], dtype=np.float32),
                },
            )[0]
            if not np.allclose(result, [4, 6]):
                return False, f"unexpected probe result: {result!r}"
            return True, f"active sessions: {active}"
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def check_gpu_support() -> bool:
    print("=" * 60)
    print("Magic-Mirror GPU acceleration check")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[FAIL] onnxruntime is not installed")
        print("       Install the Windows dependency set first.")
        return False

    print(f"onnxruntime: {ort.__version__}")
    if hasattr(ort, "preload_dlls"):
        try:
            # Helps onnxruntime-gpu find CUDA/cuDNN wheels installed by pip.
            ort.preload_dlls()
        except Exception as error:
            print(f"[WARN] CUDA DLL preload failed: {error}")

    available = ort.get_available_providers()
    print(f"registered providers: {', '.join(available)}")

    candidates = [
        ("DmlExecutionProvider", "DirectML"),
        ("CUDAExecutionProvider", "CUDA"),
    ]
    verified = []
    for provider, label in candidates:
        if provider not in available:
            print(f"[NO] {label}: provider is not registered")
            continue
        ok, detail = _probe_provider(ort, provider)
        if ok:
            verified.append(label)
            print(f"[OK] {label}: real ONNX session initialized ({detail})")
        else:
            print(f"[FAIL] {label}: registered but unusable ({detail})")

    print("=" * 60)
    if verified:
        print(f"GPU acceleration is ready: {', '.join(verified)}")
        preferred = "DirectML" if "DirectML" in verified else verified[0]
        print(
            f"For the Windows desktop app, select {preferred} "
            "in the video mode prompt."
        )
        return True

    print("No usable GPU provider was verified; processing will fall back to CPU.")
    print("Recommended Windows setup: pip install onnxruntime-directml")
    return False


if __name__ == "__main__":
    try:
        sys.exit(0 if check_gpu_support() else 1)
    except KeyboardInterrupt:
        sys.exit(1)
