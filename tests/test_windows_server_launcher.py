"""Static regressions for the packaged Windows server launcher."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / 'scripts' / 'build-windows-server.ps1'


def test_launcher_does_not_kill_every_server_exe_process():
    source = BUILD_SCRIPT.read_text(encoding='utf-8')

    assert 'taskkill /f /im server.exe' not in source
    assert 'server.pid' in source
    assert 'Win32_Process' in source
    assert 'ExecutablePath' in source

