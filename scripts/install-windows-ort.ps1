[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("directml", "cuda")]
    [string]$Runtime,

    [string]$Version = "",

    [switch]$BundleCudaDependencies
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($Version)) {
    # DirectML 1.23.0 is the latest Windows wheel; CUDA 1.27.0 is required
    # for current NVIDIA architectures such as RTX 50-series (Blackwell).
    $Version = if ($Runtime -eq "directml") { "1.23.0" } else { "1.27.0" }
}

# TinyFace pulls the CPU onnxruntime wheel as a transitive dependency. All ORT
# wheels install the same Python package, so remove every variant first.
python -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
if ($LASTEXITCODE -ne 0) {
    throw "Failed to remove conflicting ONNX Runtime distributions"
}

if ($Runtime -eq "directml") {
    $package = "onnxruntime-directml==$Version"
} else {
    $package = "onnxruntime-gpu==$Version"
}

python -m pip install --no-cache-dir --force-reinstall $package
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install $package"
}

if ($Runtime -eq "cuda" -and $BundleCudaDependencies) {
    # Pin a CUDA 12.9 / cuDNN 9 set matching ORT 1.27.0. Do not use
    # ORT's full [cuda,cudnn] extras here: TinyFace does not need NVRTC or
    # cuRAND, and including them would push the GitHub release asset close to
    # the 2 GiB per-file limit.
    $cudaPackages = @(
        "nvidia-cuda-runtime-cu12==12.9.79",
        "nvidia-cublas-cu12==12.9.2.10",
        "nvidia-cufft-cu12==11.4.1.4",
        "nvidia-cudnn-cu12==9.23.2.1"
    )
    python -m pip install --no-cache-dir --force-reinstall @cudaPackages
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install bundled CUDA runtime dependencies"
    }
}

python scripts/verify-onnxruntime.py $Runtime
if ($LASTEXITCODE -ne 0) {
    throw "ONNX Runtime verification failed for $Runtime"
}
