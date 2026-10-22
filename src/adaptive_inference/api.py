"""
Serve validated CPU predictions through a bounded inference queue.

Load and warm up the model during startup. Execute predictions through
one dedicated worker, reject excess waiting requests, and translate
queue errors into stable HTTP responses.

The initial model is untrained and provides a synthetic serving workload.
"""

import math
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from adaptive_inference.model import (
    INPUT_FEATURES,
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
    """Require an inference queue that is accepting submissions."""
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
) -> FastAPI:
    """Create a service with explicit queue and deadline settings."""
    if type(queue_capacity) is not int or queue_capacity < 1:
        raise ValueError("queue_capacity must be a positive integer")

    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or not math.isfinite(request_timeout)
        or request_timeout <= 0
    ):
        raise ValueError("request_timeout must be finite and positive")

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

        def process_features(features: list[float]) -> list[float]:
            """Run tensor conversion and inference in the queue's thread."""
            tensor = torch.tensor(
                [features],
                dtype=torch.float32,
                device="cpu",
            )

            scores = model.predict(tensor)

            return scores[0].tolist()

        scheduler = BoundedInferenceQueue(
            process_features,
            capacity=queue_capacity,
        )
        await scheduler.start()

        app.state.model = model
        app.state.scheduler = scheduler

        try:
            yield
        finally:
            # Stop admission and finish accepted work before releasing
            # the model reference used by the processor.
            try:
                await scheduler.close()
            finally:
                app.state.scheduler = None
                app.state.model = None

    app = FastAPI(
        title="Adaptive ML Inference Service",
        version="0.1.0",
        description=(
            "Bounded CPU inference with request deadlines. "
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
        """Submit validated features without blocking the event loop."""
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