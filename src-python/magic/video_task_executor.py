"""Bounded background execution and cooperative cancellation for video tasks."""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional


class VideoTaskCancelled(RuntimeError):
    """Raised when a queued or running video task is cancelled."""

    def __init__(self) -> None:
        super().__init__('cancelled')


@dataclass
class _QueuedTask:
    task_id: str
    cancel_event: threading.Event
    task_callable: Callable[[threading.Event], object]
    on_completion: Callable[[object, Optional[BaseException]], None]
    on_cancel_result: Optional[Callable[[object], None]] = None


def _default_max_workers() -> int:
    raw_value = os.environ.get('MAGIC_MIRROR_MAX_VIDEO_TASKS', '1')
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


class VideoTaskExecutor:
    """Run a bounded number of video jobs and expose per-job cancel events."""

    def __init__(self, max_workers: Optional[int] = None, name_prefix='VideoTask'):
        self.max_workers = max(1, int(max_workers or _default_max_workers()))
        self._name_prefix = str(name_prefix or 'VideoTask')
        self._queue: queue.Queue[_QueuedTask] = queue.Queue()
        self._tasks: dict[str, _QueuedTask] = {}
        self._lock = threading.RLock()
        self._workers = []

        for index in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f'{self._name_prefix}Worker-{index + 1}',
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def submit(
        self,
        task_id,
        task_callable,
        on_completion,
        on_cancel_result=None,
    ) -> threading.Event:
        """Queue a task and return the event used to cooperatively cancel it."""
        normalized_id = str(task_id)
        cancel_event = threading.Event()
        queued_task = _QueuedTask(
            task_id=normalized_id,
            cancel_event=cancel_event,
            task_callable=task_callable,
            on_completion=on_completion,
            on_cancel_result=on_cancel_result,
        )
        with self._lock:
            if normalized_id in self._tasks:
                raise RuntimeError('task-already-running')
            self._tasks[normalized_id] = queued_task
        self._queue.put(queued_task)
        return cancel_event

    def cancel(self, task_id) -> bool:
        """Signal a queued or running task to stop."""
        with self._lock:
            queued_task = self._tasks.get(str(task_id))
            if queued_task is None:
                return False
            queued_task.cancel_event.set()
            return True

    def is_active(self, task_id) -> bool:
        """Return whether a task is queued or currently running."""
        with self._lock:
            return str(task_id) in self._tasks

    def _worker(self) -> None:
        while True:
            queued_task = self._queue.get()
            result = None
            error: Optional[BaseException] = None
            try:
                if queued_task.cancel_event.is_set():
                    raise VideoTaskCancelled()
                result = queued_task.task_callable(queued_task.cancel_event)
            except BaseException as exc:
                error = exc

            with self._lock:
                current = self._tasks.get(queued_task.task_id)
                cancel_won = bool(
                    current is queued_task and queued_task.cancel_event.is_set()
                )
                if current is queued_task:
                    self._tasks.pop(queued_task.task_id, None)

            if error is None and cancel_won:
                if result is not None and queued_task.on_cancel_result is not None:
                    try:
                        queued_task.on_cancel_result(result)
                    except Exception:
                        pass
                result = None
                error = VideoTaskCancelled()
            try:
                queued_task.on_completion(result, error)
            except Exception:
                # Entry points own logging because they have the relevant context.
                pass
            finally:
                self._queue.task_done()
