"""
Verify CPU inference correctness before introducing HTTP and queueing.

Cover batch equivalence, request ordering, reproducible initialization,
input validation, and inference without gradient tracking.
"""

import pytest
import torch

from adaptive_inference.model import (
    INPUT_FEATURES,
    MAX_BATCH_SIZE,
    OUTPUT_CLASSES,
    InferenceModel,
)


@pytest.fixture(scope="module")
def model() -> InferenceModel:
    """Reuse one model across tests that perform read-only inference."""
    return InferenceModel()


def make_features(batch_size: int) -> torch.Tensor:
    """Generate distinct, reproducible inputs without using global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(17)

    return torch.randn(
        batch_size,
        INPUT_FEATURES,
        generator=generator,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32, MAX_BATCH_SIZE])
def test_output_shape_and_finite_scores(model, batch_size):
    scores = model.predict(make_features(batch_size))

    assert scores.shape == (batch_size, OUTPUT_CLASSES)
    assert scores.dtype == torch.float32
    assert scores.device.type == "cpu"
    assert torch.isfinite(scores).all().item()


def test_batched_predictions_match_individual_predictions(model):
    features = make_features(16)

    batched_scores = model.predict(features)
    individual_scores = torch.cat(
        [
            model.predict(features[index:index + 1])
            for index in range(len(features))
        ],
        dim=0,
    )

    torch.testing.assert_close(
        batched_scores,
        individual_scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_predictions_follow_input_order(model):
    features = make_features(8)
    order = torch.tensor([7, 2, 0, 5, 1, 6, 4, 3])

    original_scores = model.predict(features)
    reordered_scores = model.predict(features[order])

    torch.testing.assert_close(
        reordered_scores,
        original_scores[order],
        rtol=1e-5,
        atol=1e-6,
    )


def test_repeated_predictions_agree(model):
    features = make_features(4)

    torch.testing.assert_close(
        model.predict(features),
        model.predict(features),
        rtol=0,
        atol=0,
    )


def test_separate_model_instances_agree(model):
    features = make_features(4)
    another_model = InferenceModel()

    torch.testing.assert_close(
        model.predict(features),
        another_model.predict(features),
        rtol=0,
        atol=0,
    )


def test_model_construction_preserves_cpu_rng():
    before = torch.random.get_rng_state().clone()

    InferenceModel()

    assert torch.equal(before, torch.random.get_rng_state())


def test_inference_does_not_track_gradients(model):
    features = make_features(2).requires_grad_(True)

    scores = model.predict(features)

    assert not scores.requires_grad
    assert scores.grad_fn is None


@pytest.mark.parametrize(
    "shape",
    [
        (INPUT_FEATURES,),
        (1, 1, INPUT_FEATURES),
        (0, INPUT_FEATURES),
        (MAX_BATCH_SIZE + 1, INPUT_FEATURES),
        (1, INPUT_FEATURES - 1),
        (1, INPUT_FEATURES + 1),
    ],
)
def test_invalid_shapes_are_rejected(model, shape):
    with pytest.raises(ValueError):
        model.predict(torch.zeros(shape, dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_invalid_dtypes_are_rejected(model, dtype):
    features = torch.zeros((1, INPUT_FEATURES), dtype=dtype)

    with pytest.raises(ValueError, match="float32"):
        model.predict(features)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_are_rejected(model, value):
    features = make_features(1)
    features[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        model.predict(features)


def test_non_tensor_input_is_rejected(model):
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.predict([[0.0] * INPUT_FEATURES])