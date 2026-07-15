"""Regression tests for generation-safe image task IDs."""

import threading

import web_server
from magic import app as desktop_app


def test_web_image_task_rejects_duplicate_active_id(tmp_path):
    input_path = tmp_path / 'input.png'
    input_path.write_bytes(b'image')
    web_server._register_upload('image-input', str(input_path), 'image')
    started = threading.Event()
    release = threading.Event()
    outcome = {}

    def first_task():
        started.set()
        assert release.wait(2)
        return 'first-result'

    worker = threading.Thread(
        target=lambda: outcome.setdefault(
            'first',
            web_server._run_image_task_with_upload_pin(
                'image-input', 'same-image-id', first_task
            ),
        )
    )
    worker.start()
    assert started.wait(1)

    result, error = web_server._run_image_task_with_upload_pin(
        'image-input', 'same-image-id', lambda: 'second-result'
    )
    assert result is None
    assert str(error) == 'task-already-running'

    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert 'same-image-id' not in web_server.IMAGE_TASKS


def test_desktop_image_task_rejects_duplicate_active_id():
    started = threading.Event()
    release = threading.Event()
    outcome = {}

    def first_task():
        started.set()
        assert release.wait(2)
        return 'first-result'

    worker = threading.Thread(
        target=lambda: outcome.setdefault(
            'first',
            desktop_app._run_image_task_once('same-image-id', first_task),
        )
    )
    worker.start()
    assert started.wait(1)

    result, error = desktop_app._run_image_task_once(
        'same-image-id', lambda: 'second-result'
    )
    assert result is None
    assert str(error) == 'task-already-running'

    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert 'same-image-id' not in desktop_app.IMAGE_TASKS
