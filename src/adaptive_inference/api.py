"""
Serve validated CPU predictions through a bounded batching queue.

Load and warm up one model during startup. Combine requests into
configurable batches, run one model call per batch, and route score
vectors back to their callers.

Default settings preserve single-request execution. The initial model
is untrained and provides a synthetic serving workload.
"""

import math
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from adaptive_inference.model import (
    INPUT_FEATURES,
    MAX_BATCH_SIZE,
    OUTPUT_CLASSES,
    InferenceModel,
)
from adaptive_inference.scheduler import (
    BoundedInferenceQueue,
    InferenceFailed,
    QueueBusy,
    QueueUnavailable,
    RequestTimeout,
)


MODEL_ID = "synthetic-mlp-v1"
MAX_FEATURE_MAGNITUDE = 1_000_000.0

FeatureValue = Annotated[
    float,
    Field(
        strict=True,
        allow_inf_nan=False,
        ge=-MAX_FEATURE_MAGNITUDE,
        le=MAX_FEATURE_MAGNITUDE,
    ),
]

ScoreValue = Annotated[
    float,
    Field(allow_inf_nan=False),
]


class PredictionRequest(BaseModel):
    """Require one vector of finite numerical features."""

    model_config = ConfigDict(extra="forbid")

    features: Annotated[
        list[FeatureValue],
        Field(
            min_length=INPUT_FEATURES,
            max_length=INPUT_FEATURES,
        ),
    ]


class PredictionResponse(BaseModel):
    """Return raw scores in the model's output order."""

    model_id: str
    scores: Annotated[
        list[ScoreValue],
        Field(
            min_length=OUTPUT_CLASSES,
            max_length=OUTPUT_CLASSES,
        ),
    ]


def get_model(request: Request) -> InferenceModel:
    """Check model availability before accepting a prediction."""
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="model_not_ready",
        )

    return model


def get_scheduler(request: Request) -> BoundedInferenceQueue:
    """Require a queue that is accepting submissions."""
    scheduler = getattr(request.app.state, "scheduler", None)

    if scheduler is None or not scheduler.accepting:
        raise HTTPException(
            status_code=503,
            detail="service_unavailable",
        )

    return scheduler


def create_app(
    *,
    queue_capacity: int = 32,
    request_timeout: float = 5.0,
    max_batch_size: int = 1,
    max_batch_wait: float = 0.0,
) -> FastAPI:
    """
    Create a service with explicit queue, deadline, and batch limits.

    Batch waiting is measured in seconds. Queue capacity counts waiting
    entries; the worker additionally holds at most one batch.
    """
    if type(queue_capacity) is not int or queue_capacity < 1:
        raise ValueError("queue_capacity must be a positive integer")

    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or not math.isfinite(request_timeout)
        or request_timeout <= 0
    ):
        raise ValueError("request_timeout must be finite and positive")

    if (
        type(max_batch_size) is not int
        or not 1 <= max_batch_size <= MAX_BATCH_SIZE
    ):
        raise ValueError(
            f"max_batch_size must be between 1 and {MAX_BATCH_SIZE}"
        )

    if (
        isinstance(max_batch_wait, bool)
        or not isinstance(max_batch_wait, (int, float))
        or not math.isfinite(max_batch_wait)
        or max_batch_wait < 0
    ):
        raise ValueError("max_batch_wait must be finite and nonnegative")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Warm up the model, start its queue, and drain it at shutdown."""
        app.state.model = None
        app.state.scheduler = None

        model = InferenceModel()

        model.predict(
            torch.zeros(
                (1, INPUT_FEATURES),
                dtype=torch.float32,
                device="cpu",
            )
        )

        def process_batch(
            feature_vectors: list[list[float]],
        ) -> list[list[float]]:
            """Convert the whole batch and execute one model call."""
            tensor = torch.tensor(
                feature_vectors,
                dtype=torch.float32,
                device="cpu",
            )

            scores = model.predict(tensor)

            # Preserve the row order used by the scheduler.
            return scores.tolist()

        scheduler = BoundedInferenceQueue(
            batch_processor=process_batch,
            capacity=queue_capacity,
            max_batch_size=max_batch_size,
            max_batch_wait=max_batch_wait,
        )
        await scheduler.start()

        app.state.model = model
        app.state.scheduler = scheduler

        try:
            yield
        finally:
            try:
                await scheduler.close()
            finally:
                app.state.scheduler = None
                app.state.model = None

    app = FastAPI(
        title="Adaptive ML Inference Service",
        version="0.1.0",
        description=(
            "Bounded CPU inference with dynamic batching and deadlines. "
            "The initial model is an untrained synthetic workload."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report application liveness independently of inference."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        """Require both a loaded model and an accepting queue."""
        get_model(request)
        get_scheduler(request)

        return {
            "status": "ready",
            "model_id": MODEL_ID,
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        """Submit one request and await its own score vector."""
        get_model(request)
        scheduler = get_scheduler(request)

        try:
            scores = await scheduler.submit(
                payload.features,
                timeout=request_timeout,
            )
        except QueueBusy:
            raise HTTPException(
                status_code=503,
                detail="queue_full",
                headers={"Retry-After": "1"},
            ) from None
        except QueueUnavailable:
            raise HTTPException(
                status_code=503,
                detail="service_unavailable",
            ) from None
        except RequestTimeout:
            raise HTTPException(
                status_code=504,
                detail="request_timeout",
            ) from None
        except InferenceFailed:
            raise HTTPException(
                status_code=500,
                detail="inference_failed",
            ) from None

        return PredictionResponse(
            model_id=MODEL_ID,
            scores=scores,
        )

    return app


app = create_app()