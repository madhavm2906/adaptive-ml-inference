"""
Bound waiting inference requests and optionally combine them into batches.

Single-request mode accepts a processor mapping one feature vector to scores.
Batch mode accepts a batch_processor mapping feature vectors to score vectors.

Capacity counts entries still waiting in the queue. The worker additionally
holds at most max_batch_size entries while assembling or executing a batch.

Request deadlines include waiting, batch assembly, and execution. A timeout
cannot interrupt an already-running processor thread.

All lifecycle methods must run on the same asyncio event loop.
"""

import asyncio
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


logger = logging.getLogger(__name__)

Processor = Callable[[list[float]], list[float]]
BatchProcessor = Callable[[list[list[float]]], list[list[float]]]


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
    """Keep one request's input, deadline, and result together."""

    features: tuple[float, ...]
    deadline: float
    future: asyncio.Future[list[float]]


class BoundedInferenceQueue:
    """Execute one request or batch at a time through one worker."""

    def __init__(
        self,
        processor: Processor | None = None,
        capacity: int = 32,
        *,
        batch_processor: BatchProcessor | None = None,
        max_batch_size: int = 1,
        max_batch_wait: float = 0.0,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")

        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")

        if (
            isinstance(max_batch_wait, bool)
            or not isinstance(max_batch_wait, (int, float))
            or not math.isfinite(max_batch_wait)
            or max_batch_wait < 0
        ):
            raise ValueError("max_batch_wait must be finite and nonnegative")

        # Choose exactly one execution interface.
        if (processor is None) == (batch_processor is None):
            raise ValueError(
                "provide exactly one processor or batch_processor"
            )

        if processor is not None:
            if not callable(processor):
                raise ValueError("processor must be callable")

            if max_batch_size != 1:
                raise ValueError(
                    "max_batch_size above one requires batch_processor"
                )

        if batch_processor is not None and not callable(batch_processor):
            raise ValueError("batch_processor must be callable")

        self._processor = processor
        self._batch_processor = batch_processor
        self._max_batch_size = max_batch_size
        self._max_batch_wait = float(max_batch_wait)

        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(
            maxsize=capacity
        )
        self._executor: ThreadPoolExecutor | None = None
        self._worker: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None

        self._started = False
        self._accepting = False
        self._closed = False
        self._in_flight = 0

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
        """Return waiting entries during normal operation."""
        return self._queue.qsize()

    @property
    def in_flight(self) -> int:
        """Return entries held while assembling or executing a batch."""
        return self._in_flight

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
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            raise RequestTimeout("request deadline expired") from None

    async def close(self) -> None:
        """Stop admission, finish accepted work, and release the worker."""
        if self._close_task is None:
            self._accepting = False
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())

        # A cancelled close caller must not cancel shared cleanup.
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        """Place a stop marker after accepted entries and join the worker."""
        if self._worker is None:
            return

        await self._queue.put(None)

        try:
            await self._worker
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    async def _collect(self, jobs: list[_Job]) -> bool:
        """
        Add entries until the batch fills or its collection window ends.

        Return True if the shutdown marker was consumed. Never extend
        the collection window because another request arrives.
        """
        loop = asyncio.get_running_loop()
        collection_end = loop.time() + self._max_batch_wait

        while len(jobs) < self._max_batch_size:
            live_deadlines = [
                job.deadline
                for job in jobs
                if not job.future.done()
            ]

            if not live_deadlines:
                break

            # Do not deliberately wait beyond the earliest live deadline.
            if loop.time() >= min(live_deadlines):
                break

            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if not self._accepting:
                    break

                remaining = min(
                    collection_end,
                    min(live_deadlines),
                ) - loop.time()

                if remaining <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

            if item is None:
                self._queue.task_done()
                return True

            jobs.append(item)
            self._in_flight = len(jobs)

        return False

    def _execute(self, inputs: list[list[float]]) -> list[list[float]]:
        """Run one processor call in the dedicated executor thread."""
        if self._batch_processor is not None:
            outputs = self._batch_processor(inputs)
        else:
            assert self._processor is not None
            outputs = [self._processor(inputs[0])]

        # Validate all rows before returning any result to callers.
        if (
            not isinstance(outputs, list)
            or len(outputs) != len(inputs)
            or any(not isinstance(row, list) for row in outputs)
        ):
            raise ValueError(
                "processor must return one score list per input"
            )

        return outputs

    async def _process(self, jobs: list[_Job]) -> None:
        """Filter expired entries, execute the batch, and route results."""
        loop = asyncio.get_running_loop()
        live = []

        for job in jobs:
            if job.future.done():
                continue

            if loop.time() >= job.deadline:
                job.future.set_exception(
                    RequestTimeout("request deadline expired")
                )
            else:
                live.append(job)

        if not live:
            return

        try:
            outputs = await loop.run_in_executor(
                self._executor,
                self._execute,
                [list(job.features) for job in live],
            )
        except Exception:
            logger.exception("Queued inference failed")

            for job in live:
                if job.future.done():
                    continue

                if loop.time() >= job.deadline:
                    error = RequestTimeout("request deadline expired")
                else:
                    error = InferenceFailed("processor execution failed")

                job.future.set_exception(error)

            return

        for job, scores in zip(live, outputs, strict=True):
            if job.future.done():
                continue

            if loop.time() >= job.deadline:
                job.future.set_exception(
                    RequestTimeout("request deadline expired")
                )
            else:
                job.future.set_result(scores)

    async def _run(self) -> None:
        """Assemble and execute batches serially until shutdown."""
        while True:
            first = await self._queue.get()

            if first is None:
                self._queue.task_done()
                return

            jobs = [first]
            self._in_flight = 1

            try:
                stopping = await self._collect(jobs)
                await self._process(jobs)
            finally:
                self._in_flight = 0

                for _ in jobs:
                    self._queue.task_done()

            if stopping:
                return