"""
Run a reproducible CPU neural network for serving-infrastructure tests.

The initial weights are untrained. Outputs are numerical test scores,
not meaningful class predictions or calibrated probabilities.

Accept batches of finite float32 feature vectors and return one score
vector per input while preserving request order.
"""

import torch
from torch import nn


INPUT_FEATURES = 64
HIDDEN_FEATURES = 256
OUTPUT_CLASSES = 10
MAX_BATCH_SIZE = 128
MODEL_SEED = 2026


class InferenceModel:
    """Own a CPU model and validate inputs before running inference."""

    def __init__(self) -> None:
        # Construct the model without changing the caller's CPU RNG state.
        # Model construction happens during service startup.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(MODEL_SEED)

            self._network = nn.Sequential(
                nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES),
                nn.ReLU(),
                nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES),
                nn.ReLU(),
                nn.Linear(HIDDEN_FEATURES, OUTPUT_CLASSES),
            ).to(device="cpu", dtype=torch.float32)

        self._network.eval()
        self._network.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return scores with shape [batch_size, OUTPUT_CLASSES].

        Input must be a finite CPU float32 tensor with shape
        [batch_size, INPUT_FEATURES]. Empty and oversized batches
        are rejected.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")

        if features.device.type != "cpu":
            raise ValueError("features must be on the CPU")

        if features.dtype != torch.float32:
            raise ValueError("features must use torch.float32")

        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional tensor")

        batch_size, feature_count = features.shape

        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(
                f"batch size must be between 1 and {MAX_BATCH_SIZE}"
            )

        if feature_count != INPUT_FEATURES:
            raise ValueError(
                f"each input must contain {INPUT_FEATURES} features"
            )

        if not torch.isfinite(features).all().item():
            raise ValueError("features must contain only finite values")

        scores = self._network(features)

        # Detect numerical failures instead of returning invalid results.
        if not torch.isfinite(scores).all().item():
            raise RuntimeError("model produced non-finite scores")

        return scores