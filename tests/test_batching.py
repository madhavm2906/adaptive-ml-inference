"""
Verify dynamic batch formation, result routing, deadlines, and shutdown.

Use controlled processors to inspect actual batches. Preserve cleanup
even if a test fails while a processor is deliberately blocked.
"""

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from adaptive_inference.scheduler import (
    BoundedInferenceQueue,
    InferenceFailed,
    RequestTimeout,
)


@asynccontextmanager
async def running_batch_queue(processor, **settings):
    """Start a batching queue and guarantee worker cleanup."""
    queue = BoundedInferenceQueue(
        batch_processor=processor,
        **settings,
    )
    await queue.start()

    try:
        yield queue
    finally:
        if hasattr(processor, "release"):
            processor.release.set()

        await queue.close()


async def wait_for_state(predicate):
    """Wait for an observable state with a bounded safety deadline."""
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


class RecordingProcessor:
    """Record batches and return a distinct result for each input."""

    def __init__(self):
        self.batches = []

    def __call__(self, inputs):
        self.batches.append([list(values) for values in inputs])

        return [[sum(values)] for values in inputs]


class BlockingBatchProcessor(RecordingProcessor):
    """Hold the first batch open to create predictable queue pressure."""

    def __init__(self):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()

    def __call__(self, inputs):
        outputs = super().__call__(inputs)

        if len(self.batches) == 1:
            self.loop.call_soon_threadsafe(self.entered.set)

            if not self.release.wait(timeout=10):
                raise RuntimeError("test did not release the batch")

        return outputs


@pytest.mark.asyncio
async def test_full_batch_runs_without_waiting_for_collection_deadline():
    processor = RecordingProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=3,
        max_batch_wait=10.0,
    ) as queue:
        results = await asyncio.wait_for(
            asyncio.gather(
                queue.submit([1.0]),
                queue.submit([2.0]),
                queue.submit([3.0]),
            ),
            timeout=3,
        )

        assert results == [[1.0], [2.0], [3.0]]
        assert processor.batches == [[[1.0], [2.0], [3.0]]]


@pytest.mark.asyncio
async def test_partial_batch_runs_when_waiting_window_ends():
    processor = RecordingProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=8,
        max_batch_wait=0.02,
    ) as queue:
        result = await asyncio.wait_for(
            queue.submit([7.0]),
            timeout=3,
        )

        assert result == [7.0]
        assert processor.batches == [[[7.0]]]


@pytest.mark.asyncio
async def test_batch_limit_and_result_order():
    processor = RecordingProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=4,
        max_batch_wait=0.02,
    ) as queue:
        results = await asyncio.gather(
            *(queue.submit([float(index)]) for index in range(9))
        )

        assert results == [[float(index)] for index in range(9)]
        assert [len(batch) for batch in processor.batches] == [4, 4, 1]

        flattened = [
            features
            for batch in processor.batches
            for features in batch
        ]
        assert flattened == [[float(index)] for index in range(9)]


@pytest.mark.asyncio
async def test_expired_and_cancelled_requests_are_not_executed():
    processor = BlockingBatchProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=4,
        max_batch_wait=0.02,
    ) as queue:
        first = asyncio.create_task(queue.submit([1.0]))
        await asyncio.wait_for(processor.entered.wait(), timeout=3)

        expired = asyncio.create_task(
            queue.submit([2.0], timeout=0.05)
        )
        cancelled = asyncio.create_task(queue.submit([3.0]))
        survivor = asyncio.create_task(queue.submit([4.0]))

        await wait_for_state(lambda: queue.pending == 3)

        cancelled.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cancelled

        with pytest.raises(RequestTimeout):
            await expired

        processor.release.set()

        assert await first == [1.0]
        assert await survivor == [4.0]
        assert processor.batches == [[[1.0]], [[4.0]]]


@pytest.mark.asyncio
async def test_wrong_result_count_fails_entire_batch_and_recovers():
    calls = 0

    def processor(inputs):
        nonlocal calls
        calls += 1

        if calls == 1:
            return []

        return [[sum(values)] for values in inputs]

    async with running_batch_queue(
        processor,
        max_batch_size=2,
        max_batch_wait=0.02,
    ) as queue:
        results = await asyncio.gather(
            queue.submit([1.0]),
            queue.submit([2.0]),
            return_exceptions=True,
        )

        assert all(isinstance(result, InferenceFailed) for result in results)
        assert await queue.submit([3.0]) == [3.0]


@pytest.mark.asyncio
async def test_shutdown_flushes_a_partial_batch():
    processor = RecordingProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=8,
        max_batch_wait=10.0,
    ) as queue:
        request = asyncio.create_task(queue.submit([5.0]))

        await wait_for_state(lambda: queue.in_flight == 1)

        await asyncio.wait_for(queue.close(), timeout=3)

        assert await request == [5.0]
        assert processor.batches == [[[5.0]]]
        assert queue.in_flight == 0


@pytest.mark.asyncio
async def test_cancelled_partial_batch_does_not_call_processor():
    processor = RecordingProcessor()

    async with running_batch_queue(
        processor,
        max_batch_size=8,
        max_batch_wait=10.0,
    ) as queue:
        request = asyncio.create_task(queue.submit([5.0]))

        await wait_for_state(lambda: queue.in_flight == 1)

        request.cancel()

        with pytest.raises(asyncio.CancelledError):
            await request

        await asyncio.wait_for(queue.close(), timeout=3)

        assert processor.batches == []


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected(size):
    with pytest.raises(ValueError):
        BoundedInferenceQueue(
            batch_processor=lambda inputs: inputs,
            max_batch_size=size,
        )


@pytest.mark.parametrize(
    "wait",
    [-1, True, float("nan"), float("inf")],
)
def test_invalid_batch_wait_is_rejected(wait):
    with pytest.raises(ValueError):
        BoundedInferenceQueue(
            batch_processor=lambda inputs: inputs,
            max_batch_wait=wait,
        )


def test_batching_requires_batch_processor():
    with pytest.raises(ValueError):
        BoundedInferenceQueue(
            processor=lambda features: features,
            max_batch_size=4,
        )


def test_both_processor_interfaces_are_rejected():
    with pytest.raises(ValueError):
        BoundedInferenceQueue(
            processor=lambda features: features,
            batch_processor=lambda inputs: inputs,
        )