[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("directml", "cuda")]
    [string]$Runtime,

    [ValidateSet("disable", "force")]
    [string]$ConsoleMode = "disable",

    [string]$OutputDir = "out",

    [string]$ArchiveSuffix = "",

    [switch]$CreateCompatibilityArchive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$expectedProvider = if ($Runtime -eq "directml") {
    "DmlExecutionProvider"
} else {
    "CUDAExecutionProvider"
}

$nuitkaArgs = @(
    "--standalone",
    "--assume-yes-for-downloads",
    "--mingw64",
    "--windows-console-mode=$ConsoleMode",
    "--include-package=onnx",
    "--include-package=google.protobuf",
    "--include-package=onnxruntime",
    "--include-package-data=onnxruntime",
    "--include-package=async_tasks",
    "--include-package=cv2",
    "--include-package=numpy",
    "--include-package=tinyface",
    "--include-package=bottle",
    "--include-package-data=onnx",
    "--include-data-files=src-python/models/*.onnx=models/",
    "--output-dir=$OutputDir",
    "src-python/server.py"
)

python -m nuitka @nuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed"
}

$dist = Join-Path $PWD "$OutputDir/server.dist"
if (!(Test-Path -LiteralPath $dist -PathType Container)) {
    throw "Standalone distribution not found: $dist"
}

# ONNX Runtime and OpenCV wheels are built with the MSVC runtime.
$sys32 = Join-Path $env:WINDIR "System32"
$crtDlls = @(
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "concrt140.dll",
    "vcomp140.dll"
)
foreach ($dll in $crtDlls) {
    $source = Join-Path $sys32 $dll
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $dist $dll) -Force
    }
}

# Nuitka may omit dynamically loaded execution-provider DLLs. Copy every ORT
# DLL, including DirectML.dll, rather than only providers_*.dll.
$ortCapi = python -c "import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi')"
if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $ortCapi)) {
    throw "Unable to locate the ONNX Runtime capi directory"
}
$dstCapi = Join-Path $dist "onnxruntime/capi"
New-Item -ItemType Directory -Force -Path $dstCapi | Out-Null
Get-ChildItem -LiteralPath $ortCapi -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dstCapi $_.Name) -Force
}

if ($Runtime -eq "cuda") {
    # Place CUDA/cuDNN DLLs beside the provider so the frozen process can load
    # them without a separately installed CUDA Toolkit.
    $nvidiaRoot = python -c "import nvidia; print(next(iter(nvidia.__path__)))"
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $nvidiaRoot)) {
        throw "The nvidia dependency package is missing"
    }
    $cudaDlls = @(Get-ChildItem -LiteralPath $nvidiaRoot -Recurse -Filter "*.dll")
    if ($cudaDlls.Count -eq 0) {
        throw "No CUDA dependency DLLs found under $nvidiaRoot"
    }
    foreach ($dll in $cudaDlls) {
        Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $dstCapi $dll.Name) -Force
    }
    Write-Host "Bundled CUDA dependency DLLs:" $cudaDlls.Count

    $requiredCudaDlls = @(
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "cudnn64_9.dll"
    )
    $missingCudaDlls = @(
        $requiredCudaDlls | Where-Object {
            !(Test-Path -LiteralPath (Join-Path $dstCapi $_) -PathType Leaf)
        }
    )
    if ($missingCudaDlls.Count -gt 0) {
        throw "Missing bundled CUDA DLLs: $($missingCudaDlls -join ', ')"
    }
}

$requiredRuntimeFile = if ($Runtime -eq "directml") {
    Join-Path $dstCapi "DirectML.dll"
} else {
    Join-Path $dstCapi "onnxruntime_providers_cuda.dll"
}
if (!(Test-Path -LiteralPath $requiredRuntimeFile -PathType Leaf)) {
    throw "Required runtime file is missing: $requiredRuntimeFile"
}

# Verify the frozen executable, not only the build-time Python environment.
$serverExe = Join-Path $dist "server.exe"
$checkProcess = Start-Process `
    -FilePath $serverExe `
    -ArgumentList @("--check-ort-provider", $expectedProvider) `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($checkProcess.ExitCode -ne 0) {
    throw "Frozen server does not register $expectedProvider (exit=$($checkProcess.ExitCode))"
}

$launcherPath = Join-Path $dist "start_server.bat"
$launcherLines = @(
    '@echo off',
    'setlocal enableextensions',
    'cd /d "%~dp0"',
    '',
    'taskkill /f /im server.exe >nul 2>&1',
    '',
    'for %%F in (',
    '  vcruntime140.dll',
    '  vcruntime140_1.dll',
    '  msvcp140.dll',
    '  msvcp140_1.dll',
    '  msvcp140_2.dll',
    '  msvcp140_atomic_wait.dll',
    '  concrt140.dll',
    '  vcomp140.dll',
    ') do (',
    '  if exist "%WINDIR%\System32\%%F" (',
    '    copy /y "%WINDIR%\System32\%%F" "%~dp0%%F" >nul 2>&1',
    '  )',
    ')',
    '',
    'if not exist "%~dp0server.exe" exit /b 1',
    'start "" "%~dp0server.exe"',
    'exit /b 0'
)
$launcherLines | Set-Content -LiteralPath $launcherPath -Encoding ASCII

# The desktop client uses this marker to distinguish a newly downloaded server
# from an older server.exe left by a previous release.
$versionMatch = Select-String `
    -Path "src-python/pyproject.toml" `
    -Pattern '^version\s*=\s*"([^"]+)"' `
    | Select-Object -First 1
if (!$versionMatch) {
    throw "Unable to determine server version from src-python/pyproject.toml"
}
$serverVersion = $versionMatch.Matches[0].Groups[1].Value
$versionMarker = Join-Path $dist "server-version-$serverVersion.ok"
Set-Content -LiteralPath $versionMarker -Value $serverVersion -Encoding ASCII

$archiveStem = "server_windows_x86_64_${Runtime}${ArchiveSuffix}"
$archivePath = Join-Path $PWD "$OutputDir/$archiveStem.zip"
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Compress-Archive -Path (Join-Path $dist "*") -DestinationPath $archivePath

if ($CreateCompatibilityArchive) {
    $compatibilityPath = Join-Path $PWD "$OutputDir/server_windows_x86_64.zip"
    Copy-Item -LiteralPath $archivePath -Destination $compatibilityPath -Force
}

Write-Host "Created $archivePath"
