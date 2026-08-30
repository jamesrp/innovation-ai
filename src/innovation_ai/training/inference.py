"""CPU-only batched value inference and immutable evaluator routing."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from innovation_ai.harness.policy import BatchValueEvaluator, CandidateRoute, ValuePosition
from innovation_ai.training.checkpoint import (
    CheckpointManifest,
    LoadedCheckpoint,
    PolicyDescriptor,
    assert_policy_compatible,
    load_checkpoint,
)
from innovation_ai.training.encoding import FlatObservationEncoder
from innovation_ai.training.model import ValueNetwork


@dataclass(frozen=True, slots=True)
class CpuEvaluatorConfig:
    """Runtime-only CPU evaluator settings; they are not checkpoint identity fields."""

    microbatch_size: int = 1024
    torch_num_threads: int | None = 1

    def __post_init__(self) -> None:
        if self.microbatch_size < 1:
            raise ValueError("inference microbatch_size must be positive")
        if self.torch_num_threads is not None and self.torch_num_threads < 1:
            raise ValueError("PyTorch inference thread count must be positive or None")


class CpuBatchValueEvaluator:
    """Frozen CPU implementation of the framework-free ``BatchValueEvaluator`` protocol."""

    def __init__(
        self,
        model: ValueNetwork,
        encoder: FlatObservationEncoder | None = None,
        config: CpuEvaluatorConfig | None = None,
    ) -> None:
        self.encoder = encoder or FlatObservationEncoder()
        self.config = config or CpuEvaluatorConfig()
        if model.input_dimension != self.encoder.manifest.input_dimension:
            raise ValueError("value model input dimension differs from encoder manifest")
        if self.config.torch_num_threads is not None:
            torch.set_num_threads(self.config.torch_num_threads)
        self.model = model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        self.model.requires_grad_(False)

    def evaluate(self, positions: Sequence[ValuePosition], /) -> tuple[float, ...]:
        """Encode and evaluate positions in bounded CPU microbatches.

        Equal public positions are encoded and scored once per call, then expanded back to the
        original route order. Arena determinizations often produce identical afterstates for
        actions whose public outcome is independent of hidden-card allocation, so this avoids
        repeated encoder and model work without changing candidate means.

        Outputs are checked for finite probability values before crossing the
        framework-free contract boundary.
        """

        batch = tuple(positions)
        if not batch:
            return ()
        unique_positions: list[ValuePosition] = []
        unique_index: dict[ValuePosition, int] = {}
        restore_indices: list[int] = []
        for position in batch:
            index = unique_index.get(position)
            if index is None:
                index = len(unique_positions)
                unique_positions.append(position)
                unique_index[position] = index
            restore_indices.append(index)
        encoded = self.encoder.encode_batch(tuple(unique_positions))
        unique_values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(unique_positions), self.config.microbatch_size):
                features = torch.from_numpy(encoded[start : start + self.config.microbatch_size])
                prediction = self.model.predict(features)
                for value in prediction.tolist():
                    scalar = float(value)
                    if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                        raise ValueError("value model returned a non-probability prediction")
                    unique_values.append(scalar)
        return tuple(unique_values[index] for index in restore_indices)


def evaluate_candidate_routes(
    routes: Sequence[CandidateRoute],
    positions: Sequence[ValuePosition],
    evaluators: Mapping[str, BatchValueEvaluator],
) -> tuple[float, ...]:
    """Evaluate flattened candidates by evaluator key and restore route order.

    Grouping is deliberately by an explicit opaque key rather than any action
    index, so different frozen checkpoint policies can share an actor batch.
    """

    route_batch = tuple(routes)
    position_batch = tuple(positions)
    if len(route_batch) != len(position_batch):
        raise ValueError("candidate routes and positions must have the same length")
    grouped: dict[str, list[int]] = {}
    for index, route in enumerate(route_batch):
        grouped.setdefault(route.evaluator_key, []).append(index)
    output: list[float | None] = [None] * len(route_batch)
    for key, indices in grouped.items():
        try:
            evaluator = evaluators[key]
        except KeyError as error:
            raise KeyError(f"no evaluator is registered for key {key!r}") from error
        values = evaluator.evaluate(tuple(position_batch[index] for index in indices))
        if len(values) != len(indices):
            raise ValueError(f"evaluator {key!r} returned an incorrect result count")
        for index, value in zip(indices, values, strict=True):
            scalar = float(value)
            if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError(f"evaluator {key!r} returned a non-probability value")
            output[index] = scalar
    if any(value is None for value in output):
        raise RuntimeError("candidate evaluator routing failed to fill every output")
    return tuple(float(value) for value in output if value is not None)


CheckpointLoader = Callable[[Path, FlatObservationEncoder], LoadedCheckpoint]


def _default_checkpoint_loader(
    directory: Path,
    encoder: FlatObservationEncoder,
) -> LoadedCheckpoint:
    return load_checkpoint(directory, encoder_manifest=encoder.manifest)


class FrozenEvaluatorCache:
    """Load each immutable checkpoint once and retain only evaluation-mode models.

    Multiple policy descriptors may deliberately share a checkpoint while using
    different selection settings. The cache remains keyed by checkpoint ID, not
    mutable policy process state, and verifies each descriptor on every lookup.
    """

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        encoder: FlatObservationEncoder | None = None,
        config: CpuEvaluatorConfig | None = None,
        checkpoint_loader: CheckpointLoader | None = None,
    ) -> None:
        self.checkpoint_root = Path(checkpoint_root)
        self.encoder = encoder or FlatObservationEncoder()
        self.config = config or CpuEvaluatorConfig()
        self._checkpoint_loader = checkpoint_loader or _default_checkpoint_loader
        self._evaluators: dict[str, CpuBatchValueEvaluator] = {}
        self._manifests: dict[str, CheckpointManifest] = {}
        self._lock = threading.Lock()

    @property
    def loaded_checkpoint_ids(self) -> tuple[str, ...]:
        """Return checkpoint IDs loaded by this cache in deterministic order."""

        with self._lock:
            return tuple(sorted(self._evaluators))

    def evaluator_for(self, descriptor: PolicyDescriptor) -> CpuBatchValueEvaluator:
        """Resolve ``descriptor`` to one cached, compatibility-checked evaluator."""

        checkpoint_id = descriptor.checkpoint_id
        with self._lock:
            evaluator = self._evaluators.get(checkpoint_id)
            manifest = self._manifests.get(checkpoint_id)
            if evaluator is not None and manifest is not None:
                assert_policy_compatible(descriptor, manifest)
                return evaluator
            loaded = self._checkpoint_loader(self.checkpoint_root / checkpoint_id, self.encoder)
            assert_policy_compatible(descriptor, loaded.manifest)
            evaluator = CpuBatchValueEvaluator(loaded.model, self.encoder, self.config)
            self._evaluators[checkpoint_id] = evaluator
            self._manifests[checkpoint_id] = loaded.manifest
            return evaluator

    def evaluate_routes(
        self,
        routes: Sequence[CandidateRoute],
        positions: Sequence[ValuePosition],
        descriptors_by_key: Mapping[str, PolicyDescriptor],
    ) -> tuple[float, ...]:
        """Route candidates through descriptors and cache models by checkpoint ID."""

        evaluators: dict[str, BatchValueEvaluator] = {}
        for route in routes:
            if route.evaluator_key not in evaluators:
                try:
                    descriptor = descriptors_by_key[route.evaluator_key]
                except KeyError as error:
                    raise KeyError(
                        f"no policy descriptor is registered for key {route.evaluator_key!r}"
                    ) from error
                evaluators[route.evaluator_key] = self.evaluator_for(descriptor)
        return evaluate_candidate_routes(routes, positions, evaluators)
