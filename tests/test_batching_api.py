"""
Verify dynamic batching through HTTP using the real PyTorch model.

Check actual model-call batch sizes, per-request numerical correctness,
partial-batch execution, single-request mode, and failure recovery.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
import torch

from adaptive_inference.api import MODEL_ID, create_app
from adaptive_inference.model import INPUT_FEATURES, MAX_BATCH_SIZE


@asynccontextmanager
async def running_app(**settings):
    """Run application startup and shutdown around an async HTTP client."""
    app = create_app(**settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield app, client


def make_payload(index):
    """Give each request distinct, reproducible feature values."""
    return {
        "features": [
            (index + feature_index) / 128.0
            for feature_index in range(INPUT_FEATURES)
        ]
    }


@pytest.mark.asyncio
async def test_concurrent_requests_share_real_model_calls():
    async with running_app(
        max_batch_size=8,
        max_batch_wait=0.02,
    ) as (app, client):
        payloads = [make_payload(index) for index in range(19)]

        # Compute references before recording queued model calls.
        expected = [
            app.state.model.predict(
                torch.tensor(
                    [payload["features"]],
                    dtype=torch.float32,
                )
            )[0]
            for payload in payloads
        ]

        with patch.object(
            app.state.model,
            "predict",
            wraps=app.state.model.predict,
        ) as recorded:
            responses = await asyncio.gather(
                *(
                    client.post("/predict", json=payload)
                    for payload in payloads
                )
            )

        batch_sizes = [
            call.args[0].shape[0]
            for call in recorded.call_args_list
        ]

        assert batch_sizes == [8, 8, 3]

        for response, reference in zip(
            responses,
            expected,
            strict=True,
        ):
            assert response.status_code == 200
            assert response.json()["model_id"] == MODEL_ID

            actual = torch.tensor(
                response.json()["scores"],
                dtype=torch.float32,
            )

            torch.testing.assert_close(
                actual,
                reference,
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.mark.asyncio
async def test_one_request_runs_as_a_partial_batch():
    async with running_app(
        max_batch_size=8,
        max_batch_wait=0.01,
    ) as (app, client):
        with patch.object(
            app.state.model,
            "predict",
            wraps=app.state.model.predict,
        ) as recorded:
            response = await asyncio.wait_for(
                client.post("/predict", json=make_payload(0)),
                timeout=3,
            )

        assert response.status_code == 200
        assert recorded.call_count == 1
        assert recorded.call_args.args[0].shape == (1, INPUT_FEATURES)


@pytest.mark.asyncio
async def test_default_mode_runs_one_request_per_model_call():
    async with running_app() as (app, client):
        with patch.object(
            app.state.model,
            "predict",
            wraps=app.state.model.predict,
        ) as recorded:
            responses = await asyncio.gather(
                *(
                    client.post("/predict", json=make_payload(index))
                    for index in range(4)
                )
            )

        assert all(response.status_code == 200 for response in responses)
        assert recorded.call_count == 4
        assert all(
            call.args[0].shape == (1, INPUT_FEATURES)
            for call in recorded.call_args_list
        )


@pytest.mark.asyncio
async def test_failed_batch_returns_errors_and_worker_recovers():
    async with running_app(
        max_batch_size=4,
        max_batch_wait=0.02,
    ) as (app, client):
        with patch.object(
            app.state.model,
            "predict",
            side_effect=RuntimeError("private batch diagnostic"),
        ) as recorded:
            responses = await asyncio.gather(
                *(
                    client.post("/predict", json=make_payload(index))
                    for index in range(4)
                )
            )

        assert recorded.call_count == 1

        for response in responses:
            assert response.status_code == 500
            assert response.json() == {"detail": "inference_failed"}
            assert "private batch diagnostic" not in response.text

        recovered = await client.post(
            "/predict",
            json=make_payload(5),
        )

        assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_invalid_input_never_enters_model_execution():
    async with running_app(
        max_batch_size=8,
        max_batch_wait=0.02,
    ) as (app, client):
        with patch.object(
            app.state.model,
            "predict",
            wraps=app.state.model.predict,
        ) as recorded:
            response = await client.post(
                "/predict",
                json={"features": [0.0] * (INPUT_FEATURES - 1)},
            )

        assert response.status_code == 422
        recorded.assert_not_called()


@pytest.mark.parametrize(
    "size",
    [0, -1, MAX_BATCH_SIZE + 1, True, 1.5],
)
def test_invalid_api_batch_size_is_rejected(size):
    with pytest.raises(ValueError, match="max_batch_size"):
        create_app(max_batch_size=size)


@pytest.mark.parametrize(
    "wait",
    [-1, True, float("nan"), float("inf")],
)
def test_invalid_api_batch_wait_is_rejected(wait):
    with pytest.raises(ValueError, match="max_batch_wait"):
        create_app(max_batch_wait=wait)