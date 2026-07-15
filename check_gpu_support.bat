@echo off
chcp 65001 >nul
setlocal

echo ============================================================
echo Magic-Mirror GPU acceleration check
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python was not found on PATH.
    echo        Install Python 3.10+ or run the packaged server instead.
    pause
    exit /b 1
)

python "%~dp0check_gpu_support.py"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo GPU verification failed; CPU fallback is expected.
pause
exit /b %RESULT%
