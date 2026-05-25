import os
import signal
import sys
import time
import traceback
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server


    """Check if a directory is writable."""
def _is_dir_writable(path: str) -> bool:
    try:
        if not path:
            return False
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".magic_mirror_write_probe")
        with open(probe, "a", encoding="utf-8"):
            pass
        try:
            os.remove(probe)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _boot_log_path() -> str:
    """
    Release 包默认禁用控制台（Nuitka --windows-console-mode=disable），
    导致启动阶段异常会“无输出秒退”。
    启动日志优先写入 exe 所在目录（例如 C:\\Users\\Keggin\\MagicMirror\\），
    若不可写则回退到当前工作目录，最后回退到临时目录。
    """
    import tempfile

    candidates = []

    exe_path = os.path.abspath(sys.executable or "")
    exe_dir = os.path.dirname(exe_path) if exe_path else ""
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
            return os.path.join(directory, "magic_mirror_boot.log")

    return os.path.join(tempfile.gettempdir(), "magic_mirror_boot.log")


    """Append a message to the boot log file."""
def _append_boot_log(text: str) -> None:
    try:
        path = _boot_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        # 启动日志不应影响主流程
        pass


class _GracefulWSGIServer(ThreadingMixIn, WSGIServer):
    """Thread-per-request WSGI server with graceful shutdown support."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutting_down = False

    def shutdown_graceful(self, signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        _append_boot_log(f"received {sig_name}, shutting down gracefully...")
        self._shutting_down = True
        self.shutdown()


if __name__ == "__main__":
    _append_boot_log("=== boot ===")
    _append_boot_log(f"exe={sys.executable}")
    _append_boot_log(f"cwd={os.getcwd()}")
    _append_boot_log(f"argv={sys.argv!r}")
    _append_boot_log(f"pid={os.getpid()}")

    host = os.environ.get("MIRROR_HOST", "0.0.0.0")
    port = int(os.environ.get("MIRROR_PORT", "8023"))

    try:
        from magic.app import app

        _append_boot_log("import magic.app: OK")
        _append_boot_log(f"starting threaded wsgi server on {host}:{port}")
        httpd = make_server(
            host,
            port,
            app,
            server_class=_GracefulWSGIServer,
            handler_class=WSGIRequestHandler,
        )

        # Register graceful shutdown handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, httpd.shutdown_graceful)
            except (OSError, ValueError):
                pass

        _append_boot_log(f"server ready on {host}:{port}")
        httpd.serve_forever()
    except Exception as e:
        _append_boot_log("boot failed:")
        _append_boot_log(f"error={e!r}")
        _append_boot_log(
            "".join(traceback.format_exception(type(e), e, e.__traceback__))
        )
        _append_boot_log("exit=1")
        # 双击启动时，留一点时间让文件落盘
        time.sleep(0.2)
        sys.exit(1)
