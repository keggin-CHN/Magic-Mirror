"""Regression tests for independent upload and result file lifecycles."""

import io
import os
import time
from types import SimpleNamespace

import pytest
import web_server
from fastapi import Response
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_file_registries(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, 'WEB_DATA_DIR', str(tmp_path))
    with web_server.UPLOADS_LOCK:
        web_server.UPLOADS.clear()
    with web_server.RESULTS_LOCK:
        web_server.RESULTS.clear()
    yield
    with web_server.UPLOADS_LOCK:
        web_server.UPLOADS.clear()
    with web_server.RESULTS_LOCK:
        web_server.RESULTS.clear()


def _register_upload_and_result(tmp_path):
    input_path = tmp_path / 'shared-input.jpg'
    result_path = tmp_path / 'shared-input_output.jpg'
    input_path.write_bytes(b'input')
    result_path.write_bytes(b'result')

    web_server._register_upload('shared-upload', str(input_path), 'image')
    result_id = web_server._register_result(str(result_path))
    return input_path, result_path, result_id


def test_download_cleanup_deletes_only_result_owned_file(tmp_path):
    input_path, result_path, result_id = _register_upload_and_result(tmp_path)
    result_info = web_server._get_result_info(result_id)

    assert result_info['delete_paths'] == [str(result_path)]
    web_server._cleanup_result(result_id, list(result_info['delete_paths']))

    assert input_path.exists()
    assert web_server._get_upload_path('shared-upload') == str(input_path)
    assert not result_path.exists()
    assert web_server._get_result_info(result_id) is None


def test_result_ttl_cleanup_preserves_registered_upload(tmp_path):
    input_path, result_path, result_id = _register_upload_and_result(tmp_path)
    with web_server.RESULTS_LOCK:
        expired_at = time.time() - web_server.RESULT_TTL_SECONDS - 1
        web_server.RESULTS[result_id]['createdAt'] = expired_at
        web_server.RESULTS[result_id]['lastAccessedAt'] = expired_at

    web_server._cleanup_expired_results()

    assert input_path.exists()
    assert web_server._get_upload_path('shared-upload') == str(input_path)
    assert not result_path.exists()
    assert web_server._get_result_info(result_id) is None


def test_download_keeps_result_available_for_retry(monkeypatch, tmp_path):
    input_path, result_path, result_id = _register_upload_and_result(tmp_path)
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: True)

    response = web_server.download_file(Response(), None, result_id)

    assert response.filename == result_path.name
    assert input_path.exists()
    assert result_path.exists()
    assert web_server._get_result_info(result_id) is not None


def test_download_supports_head_probe(monkeypatch, tmp_path):
    _input_path, result_path, result_id = _register_upload_and_result(tmp_path)
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: True)

    response = TestClient(web_server.app).head(f'/api/download/{result_id}')

    assert response.status_code == 200
    assert response.headers.get('cache-control') == 'no-store'
    assert result_path.exists()


def test_registered_file_responses_disable_cache(monkeypatch, tmp_path):
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    input_path = uploads_dir / 'shared-input.jpg'
    input_path.write_bytes(b'input')
    web_server._register_upload('shared-upload', str(input_path), 'image')
    monkeypatch.setattr(web_server, 'UPLOADS_DIR', str(uploads_dir))
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: True)

    response = TestClient(web_server.app).get('/api/file/shared-upload')

    assert response.status_code == 200
    assert response.headers.get('cache-control') == 'no-store'
    assert response.content == input_path.read_bytes()


def test_library_file_response_disables_cache(monkeypatch, tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    image_path = library_dir / 'face.jpg'
    image_path.write_bytes(b'library-image')
    monkeypatch.setattr(web_server, 'LIBRARY_DIR', str(library_dir))
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: True)

    response = TestClient(web_server.app).get('/api/library/face.jpg')

    assert response.status_code == 200
    assert response.headers.get('cache-control') == 'no-store'
    assert response.content == b'library-image'


def test_save_upload_streams_file_to_destination(monkeypatch, tmp_path):
    upload = SimpleNamespace(
        filename='large video.mp4',
        file=io.BytesIO(b'abcdef'),
    )
    monkeypatch.setattr(web_server, 'UPLOAD_COPY_CHUNK_BYTES', 2)

    file_id, save_path, safe_name = web_server._save_upload(
        upload,
        str(tmp_path),
        max_bytes=6,
    )

    assert file_id
    assert safe_name.endswith('.mp4')
    assert save_path.endswith(safe_name)
    assert (tmp_path / safe_name).read_bytes() == b'abcdef'


def test_save_upload_removes_partial_file_when_too_large(monkeypatch, tmp_path):
    upload = SimpleNamespace(
        filename='too-large.mp4',
        file=io.BytesIO(b'abcdef'),
    )
    monkeypatch.setattr(web_server, 'UPLOAD_COPY_CHUNK_BYTES', 2)

    with pytest.raises(RuntimeError, match='file-too-large'):
        web_server._save_upload(upload, str(tmp_path), max_bytes=5)

    assert list(tmp_path.iterdir()) == []


def test_save_upload_removes_partial_file_when_stream_fails(
    monkeypatch, tmp_path
):
    class FailingStream:
        def __init__(self):
            self.reads = 0

        def seek(self, *_args):
            return None

        def read(self, _size):
            self.reads += 1
            if self.reads == 1:
                return b'ab'
            raise OSError('stream-failed')

    upload = SimpleNamespace(filename='broken.mp4', file=FailingStream())
    monkeypatch.setattr(web_server, 'UPLOAD_COPY_CHUNK_BYTES', 2)

    with pytest.raises(RuntimeError, match='upload-save-failed') as error:
        web_server._save_upload(upload, str(tmp_path), max_bytes=10)

    assert web_server._simplify_task_error(error.value) == 'upload-save-failed'
    assert list(tmp_path.iterdir()) == []


def test_result_path_cannot_have_two_owners(tmp_path):
    result_path = tmp_path / 'owned-result.png'
    result_path.write_bytes(b'result')
    first_id = web_server._register_result(str(result_path))

    with pytest.raises(RuntimeError, match='result-path-already-registered'):
        web_server._register_result(str(result_path))

    assert web_server._get_result_info(first_id) is not None


def test_result_registration_rejects_unmanaged_path(tmp_path):
    outside_dir = tmp_path.parent / f'{tmp_path.name}-outside'
    outside_dir.mkdir()
    outside_path = outside_dir / 'outside-result.png'
    outside_path.write_bytes(b'result')
    try:
        with pytest.raises(RuntimeError, match='invalid-path'):
            web_server._register_result(str(outside_path))
    finally:
        outside_path.unlink(missing_ok=True)
        outside_dir.rmdir()


def test_safe_delete_ignores_unmanaged_path(tmp_path):
    outside_dir = tmp_path.parent / f'{tmp_path.name}-outside'
    outside_dir.mkdir()
    outside_path = outside_dir / 'keep-me.txt'
    outside_path.write_text('keep', encoding='utf-8')
    try:
        web_server._safe_delete(str(outside_path))

        assert outside_path.exists()
    finally:
        outside_path.unlink(missing_ok=True)
        outside_dir.rmdir()


def test_safe_delete_removes_managed_symlink_without_deleting_target(
    tmp_path,
):
    uploads_dir = tmp_path / 'uploads'
    outside_dir = tmp_path.parent / f'{tmp_path.name}-outside'
    uploads_dir.mkdir()
    outside_dir.mkdir()
    target_path = outside_dir / 'target.txt'
    link_path = uploads_dir / 'target-link.txt'
    target_path.write_text('target', encoding='utf-8')
    try:
        link_path.symlink_to(target_path)
    except (NotImplementedError, OSError) as error:
        target_path.unlink(missing_ok=True)
        outside_dir.rmdir()
        pytest.skip(f'symlink creation is not available: {error}')

    try:
        web_server._safe_delete(str(link_path))

        assert not link_path.exists()
        assert target_path.exists()
    finally:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        target_path.unlink(missing_ok=True)
        outside_dir.rmdir()


def test_prepare_requires_authentication(monkeypatch):
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: False)
    assert web_server.prepare(Response(), None) == {'error': 'unauthorized'}


def test_prepare_loads_models_after_auth(monkeypatch):
    monkeypatch.setattr(web_server, '_require_auth', lambda *_args: True)
    monkeypatch.setattr(web_server, 'load_models', lambda: True)
    assert web_server.prepare(Response(), None) == {'success': True}


def test_pinned_upload_is_not_removed_by_ttl_cleanup(tmp_path):
    input_path = tmp_path / 'queued-video.mp4'
    input_path.write_bytes(b'video')
    web_server._register_upload('queued-video', str(input_path), 'video')
    with web_server.UPLOADS_LOCK:
        web_server.UPLOADS['queued-video']['lastUsedAt'] = (
            time.time() - web_server.UPLOAD_TTL_SECONDS - 1
        )

    upload_pin = web_server._pin_upload('queued-video', expected_kind='video')
    assert upload_pin is not None
    web_server._cleanup_expired_uploads()

    assert input_path.exists()
    assert web_server._get_upload_path('queued-video') == str(input_path)

    upload_pin.release()
    with web_server.UPLOADS_LOCK:
        web_server.UPLOADS['queued-video']['lastUsedAt'] = (
            time.time() - web_server.UPLOAD_TTL_SECONDS - 1
        )
    web_server._cleanup_expired_uploads()

    assert not input_path.exists()
    assert web_server._get_upload_path('queued-video') is None


def test_orphan_upload_cleanup_preserves_registered_file(
    monkeypatch, tmp_path
):
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    tracked_path = uploads_dir / 'tracked.jpg'
    orphan_path = uploads_dir / 'orphan.jpg'
    tracked_path.write_bytes(b'tracked')
    orphan_path.write_bytes(b'orphan')
    expired_at = time.time() - web_server.UPLOAD_TTL_SECONDS - 1
    monkeypatch.setattr(web_server, 'UPLOADS_DIR', str(uploads_dir))
    registered_path = os.path.join(str(uploads_dir), '.', 'tracked.jpg')
    web_server._register_upload('tracked', registered_path, 'image')
    with web_server.UPLOADS_LOCK:
        web_server.UPLOADS['tracked']['lastUsedAt'] = expired_at
    monkeypatch.setattr(web_server.os.path, 'getmtime', lambda _path: expired_at)

    web_server._cleanup_orphan_upload_files()

    assert tracked_path.exists()
    assert not orphan_path.exists()


def test_upload_pin_release_is_idempotent(tmp_path):
    input_path = tmp_path / 'shared-video.mp4'
    input_path.write_bytes(b'video')
    web_server._register_upload('shared-video', str(input_path), 'video')

    first_pin = web_server._pin_upload('shared-video', expected_kind='video')
    second_pin = web_server._pin_upload('shared-video', expected_kind='video')
    assert first_pin is not None and second_pin is not None

    first_pin.release()
    first_pin.release()
    with web_server.UPLOADS_LOCK:
        assert web_server.UPLOADS['shared-video']['activeRefs'] == 1

    second_pin.release()
    with web_server.UPLOADS_LOCK:
        assert web_server.UPLOADS['shared-video']['activeRefs'] == 0


def test_video_upload_pin_releases_when_completion_is_immediate(
    monkeypatch, tmp_path
):
    input_path = tmp_path / 'fast-video.mp4'
    input_path.write_bytes(b'video')
    web_server._register_upload('fast-video', str(input_path), 'video')
    upload_pin = web_server._pin_upload('fast-video', expected_kind='video')
    assert upload_pin is not None

    def submit(_task_id, _task_callable, on_completion, **_kwargs):
        on_completion('result.mp4', None)

    monkeypatch.setattr(
        web_server,
        'VIDEO_TASK_EXECUTOR',
        SimpleNamespace(submit=submit),
    )
    web_server._run_video_task_async(
        'fast-task',
        lambda _cancel_event: None,
        lambda _result, _error: None,
        upload_pins=[upload_pin],
    )

    with web_server.UPLOADS_LOCK:
        assert web_server.UPLOADS['fast-video']['activeRefs'] == 0


def test_video_upload_pin_releases_when_submit_fails(monkeypatch, tmp_path):
    input_path = tmp_path / 'submit-failure.mp4'
    input_path.write_bytes(b'video')
    web_server._register_upload('submit-failure', str(input_path), 'video')
    upload_pin = web_server._pin_upload(
        'submit-failure', expected_kind='video'
    )
    assert upload_pin is not None

    def submit(*_args, **_kwargs):
        raise RuntimeError('submit-failed')

    monkeypatch.setattr(
        web_server,
        'VIDEO_TASK_EXECUTOR',
        SimpleNamespace(submit=submit),
    )
    with pytest.raises(RuntimeError, match='submit-failed'):
        web_server._run_video_task_async(
            'failed-task',
            lambda _cancel_event: None,
            lambda _result, _error: None,
            upload_pins=[upload_pin],
        )

    with web_server.UPLOADS_LOCK:
        assert web_server.UPLOADS['submit-failure']['activeRefs'] == 0
