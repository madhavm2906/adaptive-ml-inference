"""
Verify HTTP behavior against the real CPU inference model.

Check startup and shutdown, prediction correctness, strict input
validation, readiness failures, and controlled inference errors.
"""

from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from adaptive_inference.api import MODEL_ID, create_app
from adaptive_inference.model import INPUT_FEATURES, OUTPUT_CLASSES


@pytest.fixture
def client():
    """Start an independent application and close it after each test."""
    app = create_app()

    with TestClient(app) as test_client:
        yield test_client


def make_payload():
    """Create distinct features with a stable order."""
    return {
        "features": [
            (index - 32) / 32.0
            for index in range(INPUT_FEATURES)
        ]
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_id": MODEL_ID,
    }


def test_prediction_matches_direct_inference(client):
    payload = make_payload()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["model_id"] == MODEL_ID
    assert len(body["scores"]) == OUTPUT_CLASSES

    expected = client.app.state.model.predict(
        torch.tensor(
            [payload["features"]],
            dtype=torch.float32,
        )
    )[0]

    actual = torch.tensor(body["scores"], dtype=torch.float32)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_repeated_requests_agree(client):
    payload = make_payload()

    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


@pytest.mark.parametrize("length", [0, 63, 65, 128])
def test_wrong_feature_count_is_rejected(client, length):
    response = client.post(
        "/predict",
        json={"features": [0.0] * length},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    ["0.5", True, None, [], {}, 1_000_001.0, -1_000_001.0],
)
def test_invalid_feature_values_are_rejected(client, value):
    payload = make_payload()
    payload["features"][0] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_integer_numbers_are_accepted(client):
    response = client.post(
        "/predict",
        json={"features": [0] * INPUT_FEATURES},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"features": None},
        {"features": "invalid"},
        {"features": [0.0] * INPUT_FEATURES, "unexpected": 1},
    ],
)
def test_invalid_request_structure_is_rejected(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/predict",
        content='{"features":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_unavailable_model_returns_503(client):
    client.app.state.model = None

    readiness = client.get("/ready")
    prediction = client.post("/predict", json=make_payload())
    health = client.get("/health")

    assert readiness.status_code == 503
    assert readiness.json() == {"detail": "model_not_ready"}

    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "model_not_ready"}

    # The application remains alive even when the model is unavailable.
    assert health.status_code == 200


def test_execution_failure_returns_controlled_error(client):
    with patch.object(
        client.app.state.model,
        "predict",
        side_effect=RuntimeError("private diagnostic information"),
    ):
        response = client.post("/predict", json=make_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "inference_failed"}
    assert "private diagnostic information" not in response.text


def test_shutdown_releases_model_reference():
    app = create_app()

    with TestClient(app):
        assert app.state.model is not None

    assert app.state.model is None