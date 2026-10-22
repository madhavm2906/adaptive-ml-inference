"""
Bound pending inference requests and execute them through one worker.

The processor receives a validated feature vector and returns scores.
Queue capacity counts waiting requests; one additional request may execute.

Deadlines include queue waiting and processor execution. A timeout stops
the caller waiting, but cannot interrupt an already-running thread.

Create, start, submit, and close the queue on the same asyncio event loop.
"""

import asyncio
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


logger = logging.getLogger(__name__)

Processor = Callable[[list[float]], list[float]]


class QueueBusy(Exception):
    """The waiting queue has reached its capacity."""


class QueueUnavailable(Exception):
    """The queue has not started or is shutting down."""


class RequestTimeout(Exception):
    """The request's waiting and execution deadline expired."""


class InferenceFailed(Exception):
    """The processor failed while executing a request."""


@dataclass
class _Job:
    """Keep one request's data, deadline, and result together."""

    features: tuple[float, ...]
    deadline: float
    future: asyncio.Future[list[float]]


class BoundedInferenceQueue:
    """Serialize processor execution while bounding waiting requests."""

    def __init__(
        self,
        processor: Processor,
        capacity: int = 32,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")

        self._processor = processor
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(
            maxsize=capacity
        )
        self._executor: ThreadPoolExecutor | None = None
        self._worker: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._started = False
        self._accepting = False
        self._closed = False

    @property
    def accepting(self) -> bool:
        """Report whether new requests can be submitted."""
        return (
            self._accepting
            and self._worker is not None
            and not self._worker.done()
        )

    @property
    def pending(self) -> int:
        """Return the number of waiting entries during normal operation."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start one worker; queue instances cannot be restarted."""
        if self._started or self._closed:
            raise RuntimeError("queue cannot be started again")

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="inference",
        )
        self._worker = asyncio.create_task(
            self._run(),
            name="inference-queue-worker",
        )
        self._started = True
        self._accepting = True

    async def submit(
        self,
        features: list[float],
        timeout: float = 5.0,
    ) -> list[float]:
        """Submit validated features without waiting for queue space."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")

        if not self.accepting:
            raise QueueUnavailable("queue is not accepting requests")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()

        job = _Job(
            features=tuple(features),
            deadline=loop.time() + timeout,
            future=future,
        )

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            future.cancel()
            raise QueueBusy("waiting queue is full") from None

        try:
            # wait_for cancels the result future on timeout.
            # Caller cancellation also propagates to that future.
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            raise RequestTimeout("request deadline expired") from None

    async def close(self) -> None:
        """
        Stop admission, drain accepted work, and release the executor.

        Concurrent close calls share one shutdown task. Cancelling a
        caller waiting for close does not cancel the shutdown itself.

        Shutdown waits for any running processor call to finish.
        """
        if self._close_task is None:
            self._accepting = False
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())

        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        """Place a stop marker after accepted requests and join the worker."""
        if self._worker is None:
            return

        await self._queue.put(None)

        try:
            await self._worker
        finally:
            # The worker awaits each processor call before finishing,
            # so its executor has no running work at this point.
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    async def _run(self) -> None:
        """Process live requests in FIFO order without blocking asyncio."""
        loop = asyncio.get_running_loop()

        while True:
            job = await self._queue.get()

            try:
                if job is None:
                    return

                # Cancelled or timed-out callers no longer need execution.
                if job.future.done():
                    continue

                if loop.time() >= job.deadline:
                    job.future.set_exception(
                        RequestTimeout("request deadline expired")
                    )
                    continue

                try:
                    scores = await loop.run_in_executor(
                        self._executor,
                        self._processor,
                        list(job.features),
                    )
                except Exception:
                    logger.exception("Queued inference failed")

                    if not job.future.done():
                        job.future.set_exception(
                            InferenceFailed("processor execution failed")
                        )

                    # A failed request must not stop the worker.
                    continue

                if job.future.done():
                    continue

                if loop.time() >= job.deadline:
                    job.future.set_exception(
                        RequestTimeout("request deadline expired")
                    )
                else:
                    job.future.set_result(scores)
            finally:
                self._queue.task_done()