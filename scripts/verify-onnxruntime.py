#!/usr/bin/env python3
"""Fail fast when a Windows ONNX Runtime build has the wrong provider."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

from packaging.version import Version

RUNTIME_CONFIG = {
    "directml": {
        "distribution": "onnxruntime-directml",
        "provider": "DmlExecutionProvider",
        "minimum_version": "1.23.0",
        "required_files": ("DirectML.dll",),
    },
    "cuda": {
        "distribution": "onnxruntime-gpu",
        "provider": "CUDAExecutionProvider",
        # ORT 1.23 registers CUDA on RTX 50-series but fails at execution time
        # with cudaErrorNoKernelImageForDevice. Keep the release on 1.27+.
        "minimum_version": "1.27.0",
        "required_files": ("onnxruntime_providers_cuda.dll",),
    },
}


def _installed_ort_distributions() -> list[str]:
    names = []
    for distribution in metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower()
        if name in {
            "onnxruntime",
            "onnxruntime-directml",
            "onnxruntime-gpu",
        }:
            names.append(name)
    return sorted(set(names))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=sorted(RUNTIME_CONFIG))
    args = parser.parse_args()

    config = RUNTIME_CONFIG[args.runtime]
    installed = _installed_ort_distributions()
    expected_distribution = config["distribution"]
    if installed != [expected_distribution]:
        raise SystemExit(
            "Expected exactly one ONNX Runtime distribution "
            f"({expected_distribution}), found: {installed}"
        )

    installed_version = metadata.version(expected_distribution)
    minimum_version = config["minimum_version"]
    if Version(installed_version) < Version(minimum_version):
        raise SystemExit(
            f"{expected_distribution} {installed_version} is too old; "
            f"minimum={minimum_version}"
        )

    import onnxruntime as ort

    providers = list(ort.get_available_providers() or [])
    expected_provider = config["provider"]
    if expected_provider not in providers:
        raise SystemExit(
            f"{expected_provider} is not registered; providers={providers}"
        )

    capi_dir = Path(ort.__file__).resolve().parent / "capi"
    missing = [
        file_name
        for file_name in config["required_files"]
        if not (capi_dir / file_name).is_file()
    ]
    if missing:
        raise SystemExit(f"Missing ONNX Runtime files in {capi_dir}: {missing}")

    print(
        json.dumps(
            {
                "runtime": args.runtime,
                "distribution": expected_distribution,
                "onnxruntimeVersion": installed_version,
                "providers": providers,
                "capiDir": str(capi_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
