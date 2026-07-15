"""Tests for bounded video task execution and cooperative cancellation."""

import threading

import pytest
from magic.video_task_executor import VideoTaskCancelled, VideoTaskExecutor


def test_executor_runs_only_one_task_at_a_time():
    executor = VideoTaskExecutor(max_workers=1, name_prefix='TestSerial')
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    completed = threading.Event()
    results = []

    def first_task(cancel_event):
        first_started.set()
        assert release_first.wait(2)
        return 'first'

    def second_task(cancel_event):
        second_started.set()
        return 'second'

    def on_completion(result, error):
        results.append((result, error))
        if len(results) == 2:
            completed.set()

    executor.submit('first', first_task, on_completion)
    executor.submit('second', second_task, on_completion)

    assert first_started.wait(1)
    assert not second_started.wait(0.1)
    release_first.set()
    assert second_started.wait(1)
    assert completed.wait(1)
    assert [result for result, _ in results] == ['first', 'second']


def test_executor_cancels_a_running_task_cooperatively():
    executor = VideoTaskExecutor(max_workers=1, name_prefix='TestCancel')
    started = threading.Event()
    completed = threading.Event()
    outcome = {}

    def task(cancel_event):
        started.set()
        assert cancel_event.wait(2)
        return 'ignored'

    def on_completion(result, error):
        outcome['result'] = result
        outcome['error'] = error
        completed.set()

    executor.submit('cancel-me', task, on_completion)
    assert started.wait(1)
    assert executor.cancel('cancel-me')
    assert completed.wait(1)
    assert outcome['result'] is None
    assert isinstance(outcome['error'], VideoTaskCancelled)
    assert not executor.is_active('cancel-me')


def test_executor_rejects_duplicate_active_task_ids():
    executor = VideoTaskExecutor(max_workers=1, name_prefix='TestDuplicate')
    release = threading.Event()

    executor.submit('same-id', lambda cancel_event: release.wait(2), lambda *_: None)
    with pytest.raises(RuntimeError, match='task-already-running'):
        executor.submit('same-id', lambda cancel_event: None, lambda *_: None)
    release.set()


def test_cancelled_result_is_cleaned_when_cancel_wins_finish_race():
    executor = VideoTaskExecutor(max_workers=1, name_prefix='TestCancelCleanup')
    started = threading.Event()
    completed = threading.Event()
    cleaned = []
    outcome = {}

    def task(cancel_event):
        started.set()
        assert cancel_event.wait(2)
        return 'completed-output.mp4'

    def on_completion(result, error):
        outcome['result'] = result
        outcome['error'] = error
        completed.set()

    executor.submit(
        'cancel-cleanup',
        task,
        on_completion,
        on_cancel_result=cleaned.append,
    )
    assert started.wait(1)
    assert executor.cancel('cancel-cleanup')
    assert completed.wait(1)
    assert cleaned == ['completed-output.mp4']
    assert outcome['result'] is None
    assert isinstance(outcome['error'], VideoTaskCancelled)


def test_task_is_not_cancellable_while_completion_callback_runs():
    executor = VideoTaskExecutor(max_workers=1, name_prefix='TestCompletionRace')
    completion_started = threading.Event()
    release_completion = threading.Event()

    def on_completion(result, error):
        assert result == 'success'
        assert error is None
        completion_started.set()
        assert release_completion.wait(2)

    executor.submit('finished', lambda cancel_event: 'success', on_completion)
    assert completion_started.wait(1)
    assert not executor.is_active('finished')
    assert not executor.cancel('finished')
    release_completion.set()
