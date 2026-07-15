"""Video wrappers must remove partial outputs on every failure path."""

import pytest
from magic import face


def _failing_processor(output_path):
    def fail(*args, **kwargs):
        output_path.write_bytes(b"partial-video")
        raise RuntimeError("processing-failed")

    return fail


def test_single_video_failure_removes_partial_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    face_path = tmp_path / "face.png"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")
    face_path.write_bytes(b"face")
    monkeypatch.setattr(face, "_get_output_video_path", lambda _path: str(output_path))
    monkeypatch.setattr(face, "_swap_face_video", _failing_processor(output_path))
    monkeypatch.setattr(face, "_log_error", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="processing-failed"):
        face.swap_face_video(str(input_path), str(face_path))

    assert not output_path.exists()


def test_multi_video_failure_removes_partial_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")
    monkeypatch.setattr(face, "_get_output_video_path", lambda _path: str(output_path))
    monkeypatch.setattr(
        face, "_swap_face_video_by_sources", _failing_processor(output_path)
    )
    monkeypatch.setattr(face, "_log_error", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="processing-failed"):
        face.swap_face_video_by_sources(str(input_path), [], [])

    assert not output_path.exists()


def test_deep_video_failure_removes_partial_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    face_path = tmp_path / "face.png"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")
    face_path.write_bytes(b"face")
    monkeypatch.setattr(face, "_get_output_video_path", lambda _path: str(output_path))
    monkeypatch.setattr(face, "_swap_face_video_deep", _failing_processor(output_path))
    monkeypatch.setattr(face, "_log_error", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="processing-failed"):
        face.swap_face_video_deep(str(input_path), [str(face_path)])

    assert not output_path.exists()
