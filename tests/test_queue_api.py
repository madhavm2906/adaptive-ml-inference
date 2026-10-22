"""
Verify bounded queue behavior through the HTTP application.

Hold model execution open to test overload, active and waiting
deadlines, responsive health checks, and readiness after queue shutdown.

These are in-process HTTP application tests, not network benchmarks.
"""

import asyncio
import threading
from contextlib import asynccontextmanager

import httpx
import pytest
import torch

from adaptive_inference.api import create_app
from adaptive_inference.model import INPUT_FEATURES, OUTPUT_CLASSES


class ControlledPrediction:
    """Block the first prediction until explicitly released."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, features):
        self.calls += 1

        if self.calls == 1:
            self.loop.call_soon_threadsafe(self.entered.set)

            if not self.release.wait(timeout=10):
                raise RuntimeError("test did not release prediction")

        return torch.zeros(
            (features.shape[0], OUTPUT_CLASSES),
            dtype=torch.float32,
        )


@asynccontextmanager
async def running_service(monkeypatch, request_timeout=5.0):
    """Start the service and guarantee cleanup of concurrent requests."""
    app = create_app(
        queue_capacity=1,
        request_timeout=request_timeout,
    )
    tasks = []

    async with app.router.lifespan_context(app):
        # Patch after startup so real model warmup still runs.
        controlled = ControlledPrediction()
        monkeypatch.setattr(
            app.state.model,
            "predict",
            controlled,
        )

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            try:
                yield app, client, controlled, tasks
            finally:
                controlled.release.set()

                if tasks:
                    await asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    )


def start_prediction(client, tasks):
    """Track each request so cleanup can await it even after a failure."""
    task = asyncio.create_task(
        client.post(
            "/predict",
            json={"features": [0.0] * INPUT_FEATURES},
        )
    )
    tasks.append(task)

    return task


async def wait_for_pending(app, count):
    """Wait for an observable queue state without fixed startup sleeps."""
    async with asyncio.timeout(3):
        while app.state.scheduler.pending != count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_full_queue_returns_503_and_health_stays_responsive(
    monkeypatch,
):
    async with running_service(monkeypatch) as service:
        app, client, controlled, tasks = service

        first = start_prediction(client, tasks)
        await asyncio.wait_for(controlled.entered.wait(), timeout=3)

        second = start_prediction(client, tasks)
        await wait_for_pending(app, 1)

        third = await start_prediction(client, tasks)

        assert third.status_code == 503
        assert third.json() == {"detail": "queue_full"}
        assert third.headers["Retry-After"] == "1"

        # These endpoints must respond while inference is blocked.
        health = await asyncio.wait_for(
            client.get("/health"),
            timeout=3,
        )
        ready = await asyncio.wait_for(
            client.get("/ready"),
            timeout=3,
        )

        assert health.status_code == 200
        assert ready.status_code == 200
        assert controlled.calls == 1

        controlled.release.set()

        assert (await first).status_code == 200
        assert (await second).status_code == 200
        assert controlled.calls == 2


@pytest.mark.asyncio
async def test_active_timeout_returns_504_without_freeing_worker(
    monkeypatch,
):
    async with running_service(
        monkeypatch,
        request_timeout=0.5,
    ) as service:
        app, client, controlled, tasks = service

        first = start_prediction(client, tasks)
        await asyncio.wait_for(controlled.entered.wait(), timeout=3)

        expired = await first

        assert expired.status_code == 504
        assert expired.json() == {"detail": "request_timeout"}

        second = start_prediction(client, tasks)
        await wait_for_pending(app, 1)

        third = await start_prediction(client, tasks)

        assert third.status_code == 503

        # The timed-out calculation still occupies the execution worker.
        assert controlled.calls == 1

        controlled.release.set()

        assert (await second).status_code == 200


@pytest.mark.asyncio
async def test_waiting_timeout_is_skipped_and_service_recovers(
    monkeypatch,
):
    async with running_service(
        monkeypatch,
        request_timeout=0.2,
    ) as service:
        app, client, controlled, tasks = service

        first = start_prediction(client, tasks)
        await asyncio.wait_for(controlled.entered.wait(), timeout=3)

        second = start_prediction(client, tasks)
        await wait_for_pending(app, 1)

        first_response = await first
        second_response = await second

        assert first_response.status_code == 504
        assert second_response.status_code == 504

        # Only the first request entered model execution.
        assert controlled.calls == 1

        controlled.release.set()

        # Wait until the expired queue entry has been removed.
        await wait_for_pending(app, 0)

        recovered = await start_prediction(client, tasks)

        assert recovered.status_code == 200
        assert controlled.calls == 2


@pytest.mark.asyncio
async def test_closed_queue_is_not_ready(monkeypatch):
    async with running_service(monkeypatch) as service:
        app, client, controlled, tasks = service

        await app.state.scheduler.close()

        ready = await client.get("/ready")
        prediction = await start_prediction(client, tasks)
        health = await client.get("/health")

        assert ready.status_code == 503
        assert ready.json() == {"detail": "service_unavailable"}

        assert prediction.status_code == 503
        assert prediction.json() == {"detail": "service_unavailable"}

        assert health.status_code == 200