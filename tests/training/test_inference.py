from __future__ import annotations

from pathlib import Path

import pytest
import torch

from innovation_ai.harness.policy import CandidateRoute, ValuePosition, build_current_value_position
from innovation_ai.innovation.actions import DrawAction
from innovation_ai.innovation.protocol import current_decision
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.training.checkpoint import (
    LoadedCheckpoint,
    PolicyDescriptor,
    load_checkpoint,
    save_checkpoint,
)
from innovation_ai.training.encoding import FlatObservationEncoder
from innovation_ai.training.inference import (
    CpuBatchValueEvaluator,
    CpuEvaluatorConfig,
    FrozenEvaluatorCache,
)
from innovation_ai.training.model import ValueNetwork


def _position() -> ValuePosition:
    state = build_setup_state(801)
    decision = current_decision(state)
    assert decision is not None
    return build_current_value_position(state, decision)


class _InferenceProbe(ValueNetwork):
    def __init__(self, input_dimension: int) -> None:
        super().__init__(input_dimension)
        self.inference_mode_observed = False

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.inference_mode_observed = torch.is_inference_mode_enabled()
        return super().predict(features)


def _constant_network(input_dimension: int, logit: float) -> ValueNetwork:
    network = ValueNetwork(input_dimension)
    with torch.no_grad():
        network.hidden.weight.zero_()
        network.hidden.bias.zero_()
        network.output.weight.zero_()
        network.output.bias.fill_(logit)
    return network


def test_cpu_evaluator_microbatches_in_inference_mode_and_matches_scalar_calls() -> None:
    encoder = FlatObservationEncoder()
    model = _InferenceProbe(encoder.manifest.input_dimension)
    evaluator = CpuBatchValueEvaluator(
        model,
        encoder,
        CpuEvaluatorConfig(microbatch_size=1, torch_num_threads=1),
    )
    positions = (_position(), _position(), _position())

    batched = evaluator.evaluate(positions)
    scalar = tuple(evaluator.evaluate((position,))[0] for position in positions)

    assert model.inference_mode_observed
    assert batched == pytest.approx(scalar, abs=1e-7)
    assert len(batched) == 3
    assert all(0.0 <= value <= 1.0 for value in batched)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert evaluator.evaluate(()) == ()


def test_frozen_cache_routes_two_checkpoint_policies_without_cross_contamination(
    tmp_path: Path,
) -> None:
    encoder = FlatObservationEncoder()
    low_directory = save_checkpoint(
        tmp_path,
        _constant_network(encoder.manifest.input_dimension, -1.0),
        encoder.manifest,
    )
    high_directory = save_checkpoint(
        tmp_path,
        _constant_network(encoder.manifest.input_dimension, 1.0),
        encoder.manifest,
    )
    low = PolicyDescriptor.from_checkpoint(load_checkpoint(low_directory).manifest)
    high = PolicyDescriptor.from_checkpoint(load_checkpoint(high_directory).manifest)
    calls: list[Path] = []

    def counting_loader(
        directory: Path,
        loaded_encoder: FlatObservationEncoder,
    ) -> LoadedCheckpoint:
        calls.append(directory)
        return load_checkpoint(directory, encoder_manifest=loaded_encoder.manifest)

    cache = FrozenEvaluatorCache(
        tmp_path,
        encoder=encoder,
        config=CpuEvaluatorConfig(microbatch_size=2, torch_num_threads=1),
        checkpoint_loader=counting_loader,
    )
    position = _position()
    draw = DrawAction(1)
    routes = (
        CandidateRoute("game-a", 1, draw, 0, low.policy_id),
        CandidateRoute("game-b", 1, draw, 0, high.policy_id),
        CandidateRoute("game-c", 1, draw, 1, low.policy_id),
    )
    descriptors = {low.policy_id: low, high.policy_id: high}

    values = cache.evaluate_routes(routes, (position, position, position), descriptors)
    repeated = cache.evaluate_routes(routes, (position, position, position), descriptors)

    assert values == pytest.approx(
        (
            torch.sigmoid(torch.tensor(-1.0)).item(),
            torch.sigmoid(torch.tensor(1.0)).item(),
            torch.sigmoid(torch.tensor(-1.0)).item(),
        ),
        abs=1e-7,
    )
    assert repeated == pytest.approx(values, abs=1e-7)
    assert calls == [low_directory, high_directory]
    assert cache.loaded_checkpoint_ids == tuple(sorted((low.checkpoint_id, high.checkpoint_id)))
