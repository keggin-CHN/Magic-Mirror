@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo Magic-Mirror GPU加速支持检查工具
echo ============================================================
echo.

REM 检查 Python 是否安装
echo [步骤 1] 检查 Python 是否安装...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装
    echo    请从 https://www.python.org/downloads/ 下载并安装 Python 3.8 或更高版本
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
if "!PYTHON_VERSION!"=="" (
    echo ❌ 无法获取 Python 版本信息
    pause
    exit /b 1
)
echo ✅ !PYTHON_VERSION!
echo.

REM 检查 onnxruntime 是否安装
echo [步骤 2] 检查 onnxruntime 是否安装...
python -c "import onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo ❌ onnxruntime 未安装
    echo    请运行: pip install onnxruntime
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python -c "import onnxruntime; print(onnxruntime.__version__)" 2^>^&1') do set ORT_VERSION=%%i
if "!ORT_VERSION!"=="" (
    echo ❌ 无法获取 onnxruntime 版本信息
    pause
    exit /b 1
)
echo ✅ onnxruntime 已安装，版本: !ORT_VERSION!
echo.

REM 检查可用的 ExecutionProvider
echo [步骤 3] 检查可用的 ExecutionProvider...
python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('PROVIDERS:' + ','.join(providers))" >nul 2>&1
if errorlevel 1 (
    echo ❌ 无法获取 ExecutionProvider
    pause
    exit /b 1
)

for /f "tokens=2 delims=:" %%i in ('python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('PROVIDERS:' + ','.join(providers))" 2^>^&1') do set PROVIDERS=%%i
if "!PROVIDERS!"=="" (
    echo ❌ 无法解析 ExecutionProvider 列表
    pause
    exit /b 1
)
echo ✅ 检测到 ExecutionProvider:
for %%p in (!PROVIDERS:,= !) do (
    echo    - %%p
)
echo.

REM 检查 GPU 加速支持
echo [步骤 4] 检查 GPU 加速支持...
set HAS_GPU=0
set GPU_TYPE=

echo !PROVIDERS! | findstr /C:"DmlExecutionProvider" >nul
if not errorlevel 1 (
    echo ✅ 支持 DirectML ^(Windows 通用 GPU 加速^)
    set HAS_GPU=1
    set GPU_TYPE=DirectML
) else (
    echo ❌ 不支持 DirectML
    echo    DirectML 需要:
    echo    - Windows 10 版本 1903 或更高
    echo    - 安装 onnxruntime-directml: pip install onnxruntime-directml
)
echo.

echo !PROVIDERS! | findstr /C:"CUDAExecutionProvider" >nul
if not errorlevel 1 (
    echo ✅ 支持 CUDA ^(NVIDIA GPU 加速^)
    set HAS_GPU=1
    if "!GPU_TYPE!"=="" (
        set GPU_TYPE=CUDA
    ) else (
        set GPU_TYPE=!GPU_TYPE! + CUDA
    )
) else (
    echo ❌ 不支持 CUDA
    echo    CUDA 需要:
    echo    - NVIDIA GPU
    echo    - 安装 CUDA Toolkit
    echo    - 安装 onnxruntime-gpu: pip install onnxruntime-gpu
)
echo.

REM 总结
echo ============================================================
echo [检查结果总结]
echo ============================================================
echo.

if !HAS_GPU! equ 1 (
    echo ✅ 系统支持 GPU 加速 ^(!GPU_TYPE!^)
    echo.
    echo 使用方法:
    echo   在视频换脸时，会弹出对话框询问是否启用GPU加速
    echo   点击"确定"启用GPU，或"取消"使用CPU
    echo.
    echo 注意事项:
    echo   - GPU 模式使用 2 个处理线程
    echo   - 如果 GPU 初始化失败，会自动回退到 CPU 模式
    echo   - 确保所有 ONNX 模型文件存在于 models 目录
) else (
    echo ❌ 系统不支持 GPU 加速，将使用 CPU 模式
    echo.
    echo 如何启用 GPU 加速:
    echo.
    echo [Windows 用户 - 推荐使用 DirectML]
    echo   1. 确保 Windows 10 版本 1903 或更高
    echo   2. 卸载现有 onnxruntime:
    echo      pip uninstall onnxruntime onnxruntime-gpu -y
    echo   3. 安装 onnxruntime-directml:
    echo      pip install onnxruntime-directml
    echo   4. 重新运行此脚本验证: check_gpu_support.bat
    echo.
    echo [NVIDIA GPU 用户 - 使用 CUDA]
    echo   1. 安装 CUDA Toolkit ^(11.x 或 12.x^)
    echo   2. 安装 cuDNN
    echo   3. 卸载现有 onnxruntime:
    echo      pip uninstall onnxruntime onnxruntime-directml -y
    echo   4. 安装 onnxruntime-gpu:
    echo      pip install onnxruntime-gpu
    echo   5. 重新运行此脚本验证: check_gpu_support.bat
)
echo ============================================================

:end
echo.
pause
