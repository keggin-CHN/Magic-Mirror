import os
import sys
import time
import traceback
from collections.abc import Sequence


def _is_dir_writable(path: str) -> bool:
    """Check if a directory is writable."""
    try:
        if not path:
            return False
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.magic_mirror_write_probe')
        with open(probe, 'a', encoding='utf-8'):
            pass
        try:
            os.remove(probe)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _boot_log_path() -> str:
    """Return a writable boot log path for frozen no-console releases."""
    import tempfile

    candidates = []

    exe_path = os.path.abspath(sys.executable or '')
    exe_dir = os.path.dirname(exe_path) if exe_path else ''
    if exe_dir:
        candidates.append(exe_dir)

    cwd = os.getcwd()
    if cwd:
        candidates.append(cwd)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        candidates.append(script_dir)

    temp_dir = tempfile.gettempdir()
    if temp_dir:
        candidates.append(temp_dir)

    for directory in candidates:
        if _is_dir_writable(directory):
            return os.path.join(directory, 'magic_mirror_boot.log')

    return os.path.join(tempfile.gettempdir(), 'magic_mirror_boot.log')


def _append_boot_log(text: str) -> None:
    """Append a message to the boot log file."""
    try:
        path = _boot_log_path()
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text)
            if not text.endswith('\n'):
                f.write('\n')
    except Exception:
        # Boot logging must never prevent the main process from continuing.
        pass


def _check_ort_provider_from_cli(argv: Sequence[str] | None = None) -> int | None:
    """Validate a provider embedded in the frozen executable for release CI."""
    args = list(sys.argv if argv is None else argv)
    flag = '--check-ort-provider'
    if flag not in args:
        return None

    try:
        index = args.index(flag)
        expected = args[index + 1]
    except (ValueError, IndexError):
        _append_boot_log(f'{flag} requires a provider name')
        return 2

    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers() or [])
        _append_boot_log(
            f'onnxruntime={ort.__version__}, providers={providers}, expected={expected}'
        )
        return 0 if expected in providers else 3
    except Exception as error:
        _append_boot_log(f'ONNX Runtime provider check failed: {error!r}')
        return 4


def _parse_port(env_name: str, default: int) -> int:
    raw_value = os.environ.get(env_name, str(default))
    try:
        port = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{env_name} must be an integer, got {raw_value!r}'
        ) from error
    if not 1 <= port <= 65535:
        raise ValueError(f'{env_name} must be between 1 and 65535, got {port}')
    return port


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    provider_check_exit = _check_ort_provider_from_cli(args)
    if provider_check_exit is not None:
        return provider_check_exit

    _append_boot_log('=== boot ===')
    _append_boot_log(f'exe={sys.executable}')
    _append_boot_log(f'cwd={os.getcwd()}')
    _append_boot_log(f'argv={args!r}')
    _append_boot_log(f'pid={os.getpid()}')

    try:
        host = os.environ.get('MIRROR_HOST', '0.0.0.0')
        port = _parse_port('MIRROR_PORT', 8023)

        import uvicorn
        from magic.app import app

        _append_boot_log('import magic.app: OK')
        _append_boot_log(f'starting ASGI server on {host}:{port}')
        uvicorn.run(app, host=host, port=port, log_config=None)
    except Exception as e:
        _append_boot_log('boot failed:')
        _append_boot_log(f'error={e!r}')
        _append_boot_log(
            ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        )
        _append_boot_log('exit=1')
        # Give double-click launches a moment to flush the boot log.
        time.sleep(0.2)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
