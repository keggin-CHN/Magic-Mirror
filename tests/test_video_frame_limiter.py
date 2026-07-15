"""Regression tests for bounded in-flight video frames."""

import threading
import time
from contextlib import nullcontext

import numpy as np
from magic import face


class _FakeCapture:
    def __init__(self, frames, expected_inflight_limit):
        self._frames = frames
        self._index = 0
        self._lock = threading.Lock()
        self.read_count = 0
        self.limit_reached = threading.Event()
        self.expected_inflight_limit = expected_inflight_limit

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == face.cv2.CAP_PROP_FPS:
            return 25.0
        if prop == face.cv2.CAP_PROP_FRAME_WIDTH:
            return 1
        if prop == face.cv2.CAP_PROP_FRAME_HEIGHT:
            return 1
        if prop == face.cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self):
        with self._lock:
            if self._index >= len(self._frames):
                return False, None
            frame = self._frames[self._index]
            self._index += 1
            self.read_count += 1
            if self.read_count >= self.expected_inflight_limit:
                self.limit_reached.set()
            return True, frame.copy()

    def set(self, *_args):
        return True

    def release(self):
        return None


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self._lock = threading.Lock()

    def isOpened(self):
        return True

    def write(self, frame):
        with self._lock:
            self.frames.append(int(frame[0, 0, 0]))

    def release(self):
        return None


class _FailingWriter(_FakeWriter):
    def write(self, _frame):
        raise RuntimeError('forced-writer-failure')


class _FakeTinyFace:
    def get_one_face(self, _image):
        return object()

    def swap_face(self, vision_frame, **_kwargs):
        return vision_frame


def test_slow_first_frame_bounds_the_entire_video_pipeline(monkeypatch, tmp_path):
    cpu_count = 3
    num_workers = 2
    queue_size = 5
    expected_inflight_limit = queue_size + num_workers
    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(40)]
    capture = _FakeCapture(frames, expected_inflight_limit)
    writer = _FakeWriter()
    slow_frame_started = threading.Event()
    release_slow_frame = threading.Event()
    errors = []

    monkeypatch.setattr(face.multiprocessing, 'cpu_count', lambda: cpu_count)
    monkeypatch.setattr(face.cv2, 'VideoCapture', lambda _path: capture)
    monkeypatch.setattr(face.cv2, 'VideoWriter', lambda *_args: writer)
    monkeypatch.setattr(face.cv2, 'VideoWriter_fourcc', lambda *_args: 0)
    monkeypatch.setattr(face, '_resolve_total_frames', lambda _path, _fps, total: total)
    monkeypatch.setattr(face, '_read_image', lambda _path: frames[0])
    monkeypatch.setattr(
        face,
        '_get_tf_pool',
        lambda **_kwargs: ([(_FakeTinyFace(), threading.RLock())], False, 'cpu'),
    )
    monkeypatch.setattr(
        face,
        '_enhanced_face_config',
        lambda *_args, **_kwargs: nullcontext(),
    )

    def fake_get_face(_tinyface, _lock, frame):
        if int(frame[0, 0, 0]) == 0:
            slow_frame_started.set()
            release_slow_frame.wait(5)
        return object()

    monkeypatch.setattr(face, '_get_face_with_retry', fake_get_face)

    def run_swap():
        try:
            face._swap_face_video(
                str(tmp_path / 'input.mp4'),
                str(tmp_path / 'face.png'),
                str(tmp_path / 'output.mp4'),
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_swap, daemon=True)
    runner.start()
    try:
        assert slow_frame_started.wait(2)
        assert capture.limit_reached.wait(2)
        time.sleep(0.2)
        with capture._lock:
            assert capture.read_count == expected_inflight_limit
    finally:
        release_slow_frame.set()

    runner.join(5)
    assert not runner.is_alive()
    assert errors == []
    assert writer.frames == list(range(len(frames)))


def test_writer_failure_stops_pipeline_without_deadlock(monkeypatch, tmp_path):
    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(20)]
    capture = _FakeCapture(frames, expected_inflight_limit=7)
    writer = _FailingWriter()
    errors = []

    monkeypatch.setattr(face.multiprocessing, 'cpu_count', lambda: 3)
    monkeypatch.setattr(face.cv2, 'VideoCapture', lambda _path: capture)
    monkeypatch.setattr(face.cv2, 'VideoWriter', lambda *_args: writer)
    monkeypatch.setattr(face.cv2, 'VideoWriter_fourcc', lambda *_args: 0)
    monkeypatch.setattr(face, '_resolve_total_frames', lambda _path, _fps, total: total)
    monkeypatch.setattr(face, '_read_image', lambda _path: frames[0])
    monkeypatch.setattr(
        face,
        '_get_tf_pool',
        lambda **_kwargs: ([(_FakeTinyFace(), threading.RLock())], False, 'cpu'),
    )
    monkeypatch.setattr(
        face,
        '_get_face_with_retry',
        lambda _tinyface, _lock, _frame: object(),
    )
    monkeypatch.setattr(
        face,
        '_enhanced_face_config',
        lambda *_args, **_kwargs: nullcontext(),
    )

    def run_swap():
        try:
            face._swap_face_video(
                str(tmp_path / 'input.mp4'),
                str(tmp_path / 'face.png'),
                str(tmp_path / 'output.mp4'),
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_swap, daemon=True)
    runner.start()
    runner.join(5)

    assert not runner.is_alive()
    assert len(errors) == 1
    assert str(errors[0]) == 'forced-writer-failure'
