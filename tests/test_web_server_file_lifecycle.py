"""Regression tests for independent upload and result file lifecycles."""

import time
from types import SimpleNamespace

import pytest
import web_server


@pytest.fixture(autouse=True)
def clear_file_registries():
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
    monkeypatch.setattr(web_server, '_require_auth', lambda: True)
    monkeypatch.setattr(
        web_server,
        'static_file',
        lambda *args, **kwargs: {'args': args, 'kwargs': kwargs},
    )

    response = web_server.download_file(result_id)

    assert response['kwargs']['download'] == result_path.name
    assert input_path.exists()
    assert result_path.exists()
    assert web_server._get_result_info(result_id) is not None


def test_result_path_cannot_have_two_owners(tmp_path):
    result_path = tmp_path / 'owned-result.png'
    result_path.write_bytes(b'result')
    first_id = web_server._register_result(str(result_path))

    with pytest.raises(RuntimeError, match='result-path-already-registered'):
        web_server._register_result(str(result_path))

    assert web_server._get_result_info(first_id) is not None


def test_prepare_requires_authentication(monkeypatch):
    monkeypatch.setattr(web_server, '_require_auth', lambda: False)
    assert web_server.prepare() == {'error': 'unauthorized'}


def test_prepare_loads_models_after_auth(monkeypatch):
    monkeypatch.setattr(web_server, '_require_auth', lambda: True)
    monkeypatch.setattr(web_server, 'load_models', lambda: True)
    assert web_server.prepare() == {'success': True}


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
