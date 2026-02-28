import os
import sys
import time
import traceback
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server


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


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


if __name__ == "__main__":
    _append_boot_log("=== boot ===")
    _append_boot_log(f"exe={sys.executable}")
    _append_boot_log(f"cwd={os.getcwd()}")
    _append_boot_log(f"argv={sys.argv!r}")

    try:
        from magic.app import app

        _append_boot_log("import magic.app: OK")
        _append_boot_log("starting threaded wsgi server on 0.0.0.0:8023")
        httpd = make_server(
            "0.0.0.0",
            8023,
            app,
            server_class=_ThreadingWSGIServer,
            handler_class=WSGIRequestHandler,
        )
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
