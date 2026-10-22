"""
Expose the CPU inference model through a validated HTTP API.

Load and warm up one model during application startup.
Provide separate liveness and readiness endpoints.
Accept one feature vector per prediction request and return raw scores.

The initial model is untrained; its scores test the serving infrastructure.
"""

import logging
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


logger = logging.getLogger(__name__)

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
    """Require exactly one vector of finite numerical features."""

    model_config = ConfigDict(extra="forbid")

    features: Annotated[
        list[FeatureValue],
        Field(
            min_length=INPUT_FEATURES,
            max_length=INPUT_FEATURES,
        ),
    ]


class PredictionResponse(BaseModel):
    """Return raw model scores in the model's output order."""

    model_id: str
    scores: Annotated[
        list[ScoreValue],
        Field(
            min_length=OUTPUT_CLASSES,
            max_length=OUTPUT_CLASSES,
        ),
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm up the model before accepting requests."""
    app.state.model = None

    model = InferenceModel()

    # Exercise the same path used by prediction requests.
    model.predict(
        torch.zeros(
            (1, INPUT_FEATURES),
            dtype=torch.float32,
            device="cpu",
        )
    )

    app.state.model = model

    try:
        yield
    finally:
        app.state.model = None


def get_model(request: Request) -> InferenceModel:
    """Return the loaded model or report that it is unavailable."""
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="model_not_ready",
        )

    return model


def create_app() -> FastAPI:
    """Create an independent application instance for serving or tests."""
    app = FastAPI(
        title="Adaptive ML Inference Service",
        version="0.1.0",
        description=(
            "CPU inference infrastructure with a synthetic model. "
            "The initial model's scores have no trained class meaning."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report liveness independently of model availability."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        """Report readiness only when the model is available."""
        get_model(request)

        return {
            "status": "ready",
            "model_id": MODEL_ID,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        """Run one validated request through the CPU model."""
        model = get_model(request)

        try:
            features = torch.tensor(
                [payload.features],
                dtype=torch.float32,
                device="cpu",
            )

            scores = model.predict(features)

            return PredictionResponse(
                model_id=MODEL_ID,
                scores=scores[0].tolist(),
            )
        except Exception:
            # Preserve diagnostic details in server logs while returning
            # a stable error response without internal implementation data.
            logger.exception("Prediction execution failed")

            raise HTTPException(
                status_code=500,
                detail="inference_failed",
            ) from None

    return app


app = create_app()