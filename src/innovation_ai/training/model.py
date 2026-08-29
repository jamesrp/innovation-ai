"""The frozen PyTorch scalar value network for Milestone 2."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

VALUE_NETWORK_ARCHITECTURE = "value-network-d128-tanh-v1"
VALUE_NETWORK_HIDDEN_DIMENSION = 128


class ValueNetwork(nn.Module):
    """Exact ``D -> 128 -> 1`` value network with probability convenience methods.

    ``forward_logits`` is the training entry point and returns one logit per row.
    ``forward`` and ``predict`` apply sigmoid and return one probability per row.
    Inputs must be two-dimensional float32 tensors with shape ``[N, D]``.
    """

    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        if input_dimension < 1:
            raise ValueError("value-network input dimension must be positive")
        self.input_dimension = input_dimension
        self.hidden = nn.Linear(input_dimension, VALUE_NETWORK_HIDDEN_DIMENSION)
        self.output = nn.Linear(VALUE_NETWORK_HIDDEN_DIMENSION, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every linear layer with Xavier-uniform weights and zero biases."""

        for layer in (self.hidden, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("value-network features must have shape [N, D]")
        if features.shape[1] != self.input_dimension:
            raise ValueError(
                "value-network feature dimension "
                f"{features.shape[1]} does not match expected {self.input_dimension}"
            )
        if features.dtype != torch.float32:
            raise ValueError("value-network features must be float32")

    def forward_logits(self, features: Tensor) -> Tensor:
        """Return shape-``[N]`` unnormalized value logits for BCE-with-logits training."""

        self._validate_features(features)
        return cast(Tensor, self.output(torch.tanh(self.hidden(features))).squeeze(-1))

    def forward(self, features: Tensor) -> Tensor:
        """Return shape-``[N]`` sigmoid value probabilities."""

        return torch.sigmoid(self.forward_logits(features))

    def predict(self, features: Tensor) -> Tensor:
        """Alias for :meth:`forward` used by inference callers."""

        return self.forward(features)
