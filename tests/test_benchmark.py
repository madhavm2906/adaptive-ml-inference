"""
Verify benchmark correctness independently of real timing performance.

Cover numerical validation, percentile definitions, failure accounting,
bounded client concurrency, request coverage, and evidence preservation.
"""

import asyncio
import json

import httpx
import pytest

from adaptive_inference.api import MODEL_ID
from adaptive_inference.benchmark import (
    run_phase,
    summarize,
    validate_result,
    write_json,
)


EXPECTED = [0.25] * 10


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"model_id": "wrong", "scores": EXPECTED},
        {"model_id": MODEL_ID, "scores": [0.25]},
        {"model_id": MODEL_ID, "scores": [True] * 10},
        {"model_id": MODEL_ID, "scores": [float("nan")] * 10},
        {"model_id": MODEL_ID, "scores": ["0.25"] * 10},
        {"model_id": MODEL_ID, "scores": [100.0] * 10},
    ],
)
def test_invalid_predictions_are_rejected(body):
    with pytest.raises(ValueError):
        validate_result(body, EXPECTED)


def test_small_rounding_difference_is_accepted():
    scores = [0.2500001] * 10

    assert validate_result(
        {"model_id": MODEL_ID, "scores": scores},
        EXPECTED,
    ) == scores


def test_nearest_rank_p95_and_failure_accounting():
    samples = [
        {"outcome": "ok", "latency_ms": float(index)}
        for index in range(1, 21)
    ]
    samples.append({"outcome": "http_503", "latency_ms": 0.1})

    summary = summarize(samples, duration=2.0)

    assert summary["attempted"] == 21
    assert summary["successful"] == 20
    assert summary["failed"] == 1
    assert summary["failure_outcomes"] == {"http_503": 1}
    assert summary["successful_requests_per_second"] == 10.0
    assert summary["successful_latency_median_ms"] == 10.5
    assert summary["successful_latency_p95_ms"] == 19.0


def test_all_failed_run_has_no_success_latency():
    summary = summarize(
        [{"outcome": "http_504", "latency_ms": 5.0}],
        duration=1.0,
    )

    assert summary["successful_requests_per_second"] == 0
    assert summary["successful_latency_median_ms"] is None
    assert summary["successful_latency_p95_ms"] is None
    assert summary["failure_rate"] == 1.0


@pytest.mark.asyncio
async def test_client_concurrency_is_bounded_and_every_request_is_recorded():
    active = 0
    maximum_active = 0

    async def handler(request):
        nonlocal active, maximum_active

        active += 1
        maximum_active = max(maximum_active, active)

        try:
            await asyncio.sleep(0)

            return httpx.Response(
                200,
                json={"model_id": MODEL_ID, "scores": EXPECTED},
            )
        finally:
            active -= 1

    workload = [{
        "features": [0.0] * 64,
        "expected_scores": EXPECTED,
    }]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    ) as client:
        samples, duration = await run_phase(
            client,
            workload,
            count=11,
            concurrency=3,
            timeout=2,
        )

    assert duration > 0
    assert 1 <= maximum_active <= 3
    assert [row["request_id"] for row in samples] == list(range(11))
    assert all(row["outcome"] == "ok" for row in samples)


@pytest.mark.asyncio
async def test_http_validation_and_transport_failures_are_retained():
    async def handler(request):
        index = json.loads(request.content)["features"][0]

        if index == 0:
            return httpx.Response(
                200,
                json={"model_id": MODEL_ID, "scores": EXPECTED},
            )

        if index == 1:
            return httpx.Response(503, json={"detail": "queue_full"})

        if index == 2:
            return httpx.Response(
                200,
                json={"model_id": MODEL_ID, "scores": [99.0] * 10},
            )

        raise httpx.ReadTimeout("deliberate timeout", request=request)

    workload = [
        {"features": [index] * 64, "expected_scores": EXPECTED}
        for index in range(4)
    ]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    ) as client:
        samples, duration = await run_phase(
            client,
            workload,
            count=4,
            concurrency=2,
            timeout=2,
        )

    assert [row["outcome"] for row in samples] == [
        "ok",
        "http_503",
        "invalid_response",
        "transport_ReadTimeout",
    ]
    assert summarize(samples, duration)["failed"] == 3


def test_existing_evidence_is_not_overwritten(tmp_path):
    path = tmp_path / "metadata.json"
    write_json(path, {"original": True})

    with pytest.raises(FileExistsError):
        write_json(path, {"replacement": True})

    assert json.loads(path.read_text()) == {"original": True}