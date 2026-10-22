"""
Verify queue bounds, request deadlines, cancellation, and shutdown.

Use a controlled blocking processor to create predictable queue pressure.
Always release blocked work during cleanup, including when assertions fail.
"""

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from adaptive_inference.scheduler import (
    BoundedInferenceQueue,
    InferenceFailed,
    QueueBusy,
    QueueUnavailable,
    RequestTimeout,
)


class BlockingProcessor:
    """Hold the first call open until the test explicitly releases it."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.calls = []

    def __call__(self, features):
        self.calls.append(list(features))

        if len(self.calls) == 1:
            self.loop.call_soon_threadsafe(self.entered.set)

            # A safety deadline prevents a broken test hanging forever.
            if not self.release.wait(timeout=10):
                raise RuntimeError("test did not release the processor")

        return [sum(features)]


@asynccontextmanager
async def running_queue(processor, capacity=2):
    """Start a queue and guarantee cleanup after the test."""
    queue = BoundedInferenceQueue(processor, capacity=capacity)
    await queue.start()

    try:
        yield queue
    finally:
        if isinstance(processor, BlockingProcessor):
            processor.release.set()

        await queue.close()


async def wait_for_pending(queue, count):
    """Wait for an observable queue state with a bounded test deadline."""
    async with asyncio.timeout(3):
        while queue.pending != count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_result_is_returned():
    async with running_queue(lambda values: [sum(values)]) as queue:
        result = await queue.submit([1.0, 2.0, 3.0])

        assert result == [6.0]


@pytest.mark.asyncio
async def test_full_queue_rejects_without_running_extra_work():
    processor = BlockingProcessor()

    async with running_queue(processor, capacity=1) as queue:
        first = asyncio.create_task(queue.submit([1.0]))
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        second = asyncio.create_task(queue.submit([2.0]))
        await wait_for_pending(queue, 1)

        with pytest.raises(QueueBusy):
            await queue.submit([3.0])

        processor.release.set()

        assert await first == [1.0]
        assert await second == [2.0]
        assert processor.calls == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_queued_timeout_skips_processor_execution():
    processor = BlockingProcessor()

    async with running_queue(processor) as queue:
        first = asyncio.create_task(queue.submit([1.0]))
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        with pytest.raises(RequestTimeout):
            await queue.submit([2.0], timeout=0.05)

        processor.release.set()

        assert await first == [1.0]

        # This request completes after the expired entry is skipped.
        assert await queue.submit([3.0]) == [3.0]
        assert processor.calls == [[1.0], [3.0]]


@pytest.mark.asyncio
async def test_running_timeout_does_not_free_execution_worker():
    processor = BlockingProcessor()

    async with running_queue(processor, capacity=1) as queue:
        first = asyncio.create_task(
            queue.submit([1.0], timeout=0.2)
        )
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        with pytest.raises(RequestTimeout):
            await first

        second = asyncio.create_task(queue.submit([2.0]))
        await wait_for_pending(queue, 1)

        with pytest.raises(QueueBusy):
            await queue.submit([3.0])

        # The first calculation still owns the worker after its timeout.
        assert processor.calls == [[1.0]]

        processor.release.set()

        assert await second == [2.0]


@pytest.mark.asyncio
async def test_cancelled_waiting_request_is_skipped():
    processor = BlockingProcessor()

    async with running_queue(processor) as queue:
        first = asyncio.create_task(queue.submit([1.0]))
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        cancelled = asyncio.create_task(queue.submit([2.0]))
        await wait_for_pending(queue, 1)

        cancelled.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cancelled

        processor.release.set()

        assert await first == [1.0]
        assert await queue.submit([3.0]) == [3.0]
        assert processor.calls == [[1.0], [3.0]]


@pytest.mark.asyncio
async def test_processor_failure_does_not_stop_worker():
    def processor(features):
        if features == [0.0]:
            raise RuntimeError("deliberate test failure")

        return [sum(features)]

    async with running_queue(processor) as queue:
        with pytest.raises(InferenceFailed):
            await queue.submit([0.0])

        assert await queue.submit([4.0]) == [4.0]
        assert queue.accepting


@pytest.mark.asyncio
async def test_shutdown_drains_work_and_rejects_new_requests():
    processor = BlockingProcessor()

    async with running_queue(processor) as queue:
        first = asyncio.create_task(queue.submit([1.0]))
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        second = asyncio.create_task(queue.submit([2.0]))
        await wait_for_pending(queue, 1)

        closing = asyncio.create_task(queue.close())

        # Let close begin and turn off request admission.
        await asyncio.sleep(0)

        assert not queue.accepting

        with pytest.raises(QueueUnavailable):
            await queue.submit([3.0])

        processor.release.set()

        assert await first == [1.0]
        assert await second == [2.0]
        await closing

        # Repeated close calls are safe.
        await queue.close()


@pytest.mark.asyncio
async def test_submission_requires_started_queue():
    queue = BoundedInferenceQueue(lambda values: values)

    with pytest.raises(QueueUnavailable):
        await queue.submit([1.0])

    await queue.close()

    with pytest.raises(RuntimeError):
        await queue.start()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        BoundedInferenceQueue(lambda values: values, capacity=capacity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, float("nan"), float("inf")],
)
async def test_invalid_timeout_is_rejected(timeout):
    async with running_queue(lambda values: values) as queue:
        with pytest.raises(ValueError):
            await queue.submit([1.0], timeout=timeout)