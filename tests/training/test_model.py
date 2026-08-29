from __future__ import annotations

import math

import pytest
import torch

from innovation_ai.training.model import (
    VALUE_NETWORK_HIDDEN_DIMENSION,
    ValueNetwork,
)


def test_value_network_is_exact_tanh_scalar_architecture_with_zero_biases() -> None:
    network = ValueNetwork(3)

    assert network.hidden.in_features == 3
    assert network.hidden.out_features == VALUE_NETWORK_HIDDEN_DIMENSION
    assert network.output.in_features == VALUE_NETWORK_HIDDEN_DIMENSION
    assert network.output.out_features == 1
    assert torch.count_nonzero(network.hidden.bias) == 0
    assert torch.count_nonzero(network.output.bias) == 0

    with torch.no_grad():
        network.hidden.weight.zero_()
        network.output.weight.zero_()
        network.hidden.weight[0, 0] = 2.0
        network.output.weight[0, 0] = 3.0
    features = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

    logits = network.forward_logits(features)
    probabilities = network.predict(features)
    expected_logit = 3.0 * math.tanh(1.0)

    assert logits.shape == (1,)
    assert probabilities.shape == (1,)
    assert logits.item() == pytest.approx(expected_logit)
    assert probabilities.item() == pytest.approx(1.0 / (1.0 + math.exp(-expected_logit)))


def test_value_network_batch_shape_probability_range_and_input_validation() -> None:
    network = ValueNetwork(4)
    features = torch.linspace(-1.0, 1.0, 20, dtype=torch.float32).reshape(5, 4)

    probabilities = network(features)

    assert probabilities.shape == (5,)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    with pytest.raises(ValueError, match="shape"):
        network(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="dimension"):
        network(torch.zeros((1, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="float32"):
        network(torch.zeros((1, 4), dtype=torch.float64))
