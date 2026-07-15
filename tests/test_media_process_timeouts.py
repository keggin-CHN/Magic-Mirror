"""Regression tests for ffprobe/ffmpeg timeout handling."""

import subprocess

from magic import face


def test_ffprobe_timeout_falls_back_to_unknown_total(monkeypatch, tmp_path):
    monkeypatch.setattr(face, "_resolve_ffprobe_binary", lambda: "ffprobe")

    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == 15
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(face.subprocess, "run", timed_out)
    assert face._resolve_total_frames(str(tmp_path / "input.mp4"), 25, 0) == 0


def test_ffmpeg_timeout_terminates_and_removes_temp_file(monkeypatch, tmp_path):
    output_path = tmp_path / "output.mp4"
    temp_path = tmp_path / "output_mux_tmp.mp4"
    temp_path.write_bytes(b"partial")

    class HangingProcess:
        returncode = None

        def communicate(self, timeout=None):
            if timeout == 0.2:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return "", ""

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setenv("MAGIC_FFMPEG_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(face, "_resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(face.subprocess, "Popen", lambda *args, **kwargs: HangingProcess())
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(face.time, "monotonic", lambda: next(clock))

    try:
        face._try_mux_audio("input.mp4", str(output_path))
    except RuntimeError as error:
        assert str(error) == "ffmpeg-timeout"
    else:
        raise AssertionError("expected ffmpeg timeout")

    assert not temp_path.exists()
