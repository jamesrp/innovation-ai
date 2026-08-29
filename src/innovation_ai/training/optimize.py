"""Deterministic CPU terminal-outcome training for materialized value datasets.

The optimizer consumes only sealed :mod:`dataset` manifests and their immutable NPZ
shards.  It never re-splits examples: the episode-grouped held-out partition in the
manifest is the sole validation source.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from innovation_ai.training.checkpoint import save_checkpoint
from innovation_ai.training.compact_replay import sha256_digest
from innovation_ai.training.dataset import (
    DatasetManifest,
    DatasetMaterializationError,
    DatasetShard,
    DatasetSplit,
    dumps_dataset_manifest,
    load_dataset_shard,
    read_dataset_manifest,
)
from innovation_ai.training.encoding import EncoderManifest, build_encoder_manifest
from innovation_ai.training.model import ValueNetwork

DEFAULT_CALIBRATION_BIN_COUNT = 10


class TrainingError(ValueError):
    """The immutable dataset or requested CPU training run is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Strict, reproducible CPU defaults for terminal-outcome optimization."""

    seed: int = 0
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    torch_num_threads: int = 1
    calibration_bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT
    min_brier_improvement: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training seed must be a non-negative integer")
        positive_ints = (
            "max_epochs",
            "patience",
            "batch_size",
            "torch_num_threads",
            "calibration_bin_count",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"training {name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "min_brier_improvement"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"training {name} must be finite")
            if value < 0.0:
                raise ValueError(f"training {name} must be non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("training learning_rate must be positive")

    def payload(self) -> dict[str, object]:
        """Return JSON-safe resolved configuration recorded with a checkpoint."""

        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One fixed-width probability bin; empty-bin means are explicitly ``None``."""

    lower: float
    upper: float
    count: int
    mean_prediction: float | None
    mean_target: float | None

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    """Loss and calibration summary for one immutable dataset partition."""

    bce: float
    brier: float
    mean_prediction: float
    calibration: tuple[CalibrationBin, ...]

    def payload(self) -> dict[str, object]:
        return {
            "bce": self.bce,
            "brier": self.brier,
            "mean_prediction": self.mean_prediction,
            "calibration": [bin_.payload() for bin_ in self.calibration],
        }


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One completed epoch, with held-out Brier used for selection."""

    epoch: int
    train_bce: float
    validation_bce: float
    validation_brier: float
    examples_per_second: float

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedTrainingDataset:
    """Validated, contiguous arrays and provenance from a sealed dataset manifest."""

    manifest: DatasetManifest
    dataset_id: str
    train_features: NDArray[np.float32]
    train_targets: NDArray[np.float32]
    validation_features: NDArray[np.float32]
    validation_targets: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Final restored-best model metrics and complete epoch-level audit history."""

    dataset_id: str
    config: TrainingConfig
    train: SplitMetrics
    validation: SplitMetrics
    held_out_episode_count: int
    held_out_game_count: int
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    examples_per_second: float
    epoch_history: tuple[EpochRecord, ...]

    def payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "training_config": self.config.payload(),
            "train": self.train.payload(),
            "validation": self.validation.payload(),
            "held_out_episode_count": self.held_out_episode_count,
            "held_out_game_count": self.held_out_game_count,
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stopped_early,
            "examples_per_second": self.examples_per_second,
            "epoch_history": [record.payload() for record in self.epoch_history],
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The restored-best model, immutable checkpoint path, and saved metrics."""

    checkpoint_directory: Path
    model: ValueNetwork
    report: TrainingReport


def dataset_id(manifest: DatasetManifest) -> str:
    """Return the content identity of the canonical one-line dataset manifest."""

    return sha256_digest((dumps_dataset_manifest(manifest) + "\n").encode("ascii"))


def load_training_dataset(
    manifest_path: str | Path,
    *,
    encoder_manifest: EncoderManifest | None = None,
) -> LoadedTrainingDataset:
    """Load every manifest-assigned NPZ shard after digest and content validation."""

    path = Path(manifest_path)
    manifest = read_dataset_manifest(path)
    expected_encoder = encoder_manifest or build_encoder_manifest()
    if manifest.encoder_fingerprint != expected_encoder.layout_fingerprint:
        raise TrainingError("dataset encoder fingerprint differs from training encoder")
    if manifest.encoder_version != expected_encoder.encoder_version:
        raise TrainingError("dataset encoder version differs from training encoder")

    arrays_by_split: dict[DatasetSplit, list[dict[str, NDArray[np.generic]]]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
    }
    for shard in manifest.shards:
        arrays_by_split[shard.split].append(
            _load_validated_shard(path.parent, shard, expected_encoder.input_dimension)
        )

    train_features, train_targets = _join_split(arrays_by_split[DatasetSplit.TRAIN], "train")
    validation_features, validation_targets = _join_split(
        arrays_by_split[DatasetSplit.VALIDATION], "validation"
    )
    if train_features.shape[0] != manifest.counts.train_example_count:
        raise TrainingError("train NPZ examples differ from dataset manifest count")
    if validation_features.shape[0] != manifest.counts.validation_example_count:
        raise TrainingError("validation NPZ examples differ from dataset manifest count")
    if train_features.shape[0] == 0:
        raise TrainingError("terminal training requires at least one train example")
    if validation_features.shape[0] == 0:
        raise TrainingError("held-out Brier early stopping requires validation examples")
    return LoadedTrainingDataset(
        manifest,
        dataset_id(manifest),
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    )


def train_terminal_outcomes(
    manifest_path: str | Path,
    checkpoint_root: str | Path,
    *,
    config: TrainingConfig | None = None,
    encoder_manifest: EncoderManifest | None = None,
    parent_checkpoint_ids: Sequence[str] = (),
    generation: int = 0,
    creation_command: str = "innovation-ai train-value",
) -> TrainingResult:
    """Fit a CPU value model with BCE logits and held-out-Brier early stopping.

    The model and optimizer are restored to their best validation-Brier epoch before
    immutable publication.  All randomness (initialization and epoch shuffles) is
    reset from ``config.seed`` for reproducibility on fixed CPU software/hardware.
    """

    resolved = config or TrainingConfig()
    resolved_encoder = encoder_manifest or build_encoder_manifest()
    data = load_training_dataset(manifest_path, encoder_manifest=resolved_encoder)
    _configure_cpu_reproducibility(resolved)

    model = ValueNetwork(resolved_encoder.input_dimension).to(device="cpu", dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )
    loss = BCEWithLogitsLoss(reduction="mean")
    random = np.random.default_rng(resolved.seed)
    train_features = torch.from_numpy(data.train_features)
    train_targets = torch.from_numpy(data.train_targets)
    validation_features = torch.from_numpy(data.validation_features)
    validation_targets = torch.from_numpy(data.validation_targets)

    best_brier = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_model_state: dict[str, Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    history: list[EpochRecord] = []
    stopped_early = False

    for epoch in range(1, resolved.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        indices = random.permutation(train_features.shape[0])
        for start in range(0, len(indices), resolved.batch_size):
            batch_indices = torch.from_numpy(indices[start : start + resolved.batch_size])
            features = train_features.index_select(0, batch_indices)
            targets = train_targets.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(model.forward_logits(features), targets)
            batch_loss.backward()
            optimizer.step()
        elapsed = max(time.perf_counter() - started, float.fromhex("0x1.0p-1022"))
        train_metrics = _evaluate(
            model, train_features, train_targets, resolved.calibration_bin_count
        )
        validation_metrics = _evaluate(
            model, validation_features, validation_targets, resolved.calibration_bin_count
        )
        history.append(
            EpochRecord(
                epoch,
                train_metrics.bce,
                validation_metrics.bce,
                validation_metrics.brier,
                train_features.shape[0] / elapsed,
            )
        )
        if validation_metrics.brier < best_brier - resolved.min_brier_improvement:
            best_brier = validation_metrics.brier
            best_epoch = epoch
            stale_epochs = 0
            best_model_state = _clone_model_state(model)
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= resolved.patience:
                stopped_early = True
                break

    if best_model_state is None or best_optimizer_state is None:
        # Finite validated inputs always execute at least one epoch.
        raise RuntimeError("terminal optimizer did not complete an epoch")
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    final_train = _evaluate(model, train_features, train_targets, resolved.calibration_bin_count)
    final_validation = _evaluate(
        model, validation_features, validation_targets, resolved.calibration_bin_count
    )
    held_out_episodes = data.manifest.counts.validation_episode_count
    examples_per_second = (
        train_features.shape[0]
        * len(history)
        / sum(train_features.shape[0] / record.examples_per_second for record in history)
    )
    report = TrainingReport(
        data.dataset_id,
        resolved,
        final_train,
        final_validation,
        held_out_episodes,
        held_out_episodes,
        best_epoch,
        len(history),
        stopped_early,
        examples_per_second,
        tuple(history),
    )
    checkpoint_directory = save_checkpoint(
        checkpoint_root,
        model,
        resolved_encoder,
        optimizer=optimizer,
        metrics=report.payload(),
        training_dataset_ids=(data.dataset_id,),
        parent_checkpoint_ids=tuple(parent_checkpoint_ids),
        generation=generation,
        optimizer_config={
            "name": "AdamW",
            "learning_rate": resolved.learning_rate,
            "weight_decay": resolved.weight_decay,
            "batch_size": resolved.batch_size,
            "seed": resolved.seed,
            "max_epochs": resolved.max_epochs,
            "patience": resolved.patience,
            "torch_num_threads": resolved.torch_num_threads,
        },
        creation_command=creation_command,
    )
    return TrainingResult(checkpoint_directory, model, report)


# Clear aliases for callers that name this pipeline step after its model rather than its target.
train_value_model = train_terminal_outcomes
optimize_terminal_outcomes = train_terminal_outcomes


def _load_validated_shard(
    directory: Path, shard: DatasetShard, input_dimension: int
) -> dict[str, NDArray[np.generic]]:
    path = directory / f"{shard.shard_id}.npz"
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TrainingError(f"could not read dataset shard {shard.shard_id}: {error}") from error
    if sha256_digest(raw) != shard.sha256:
        raise TrainingError(f"dataset shard {shard.shard_id} digest differs from manifest")
    try:
        arrays = load_dataset_shard(path)
    except DatasetMaterializationError as error:
        raise TrainingError(f"dataset shard {shard.shard_id} is invalid: {error}") from error
    features = arrays["features"]
    if features.shape[1] != input_dimension:
        raise TrainingError(
            f"dataset shard {shard.shard_id} feature dimension differs from encoder"
        )
    if features.shape[0] != shard.example_count:
        raise TrainingError(f"dataset shard {shard.shard_id} example count differs from manifest")
    actual_episodes = set(cast(NDArray[np.str_], arrays["episode_ids"]).tolist())
    if not actual_episodes.issubset(set(shard.episode_ids)):
        raise TrainingError(f"dataset shard {shard.shard_id} contains an unassigned episode")
    return arrays


def _join_split(
    shards: Sequence[dict[str, NDArray[np.generic]]], split: str
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if not shards:
        raise TrainingError(f"dataset manifest has no {split} NPZ shards")
    features = np.concatenate(
        [cast(NDArray[np.float32], item["features"]) for item in shards], axis=0
    )
    targets = np.concatenate(
        [cast(NDArray[np.float32], item["targets"]) for item in shards], axis=0
    )
    return np.ascontiguousarray(features, dtype=np.float32), np.ascontiguousarray(
        targets, dtype=np.float32
    )


def _configure_cpu_reproducibility(config: TrainingConfig) -> None:
    """Set process-wide CPU training controls before model initialization."""

    torch.set_num_threads(config.torch_num_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)


def _clone_model_state(model: ValueNetwork) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _evaluate(
    model: ValueNetwork,
    features: Tensor,
    targets: Tensor,
    bin_count: int,
) -> SplitMetrics:
    model.eval()
    with torch.inference_mode():
        logits = model.forward_logits(features)
        bce = float(BCEWithLogitsLoss(reduction="mean")(logits, targets).item())
        predictions = torch.sigmoid(logits).cpu().numpy().astype(np.float64, copy=False)
    labels = targets.cpu().numpy().astype(np.float64, copy=False)
    brier = float(np.mean(np.square(predictions - labels), dtype=np.float64))
    return SplitMetrics(
        bce,
        brier,
        float(np.mean(predictions, dtype=np.float64)),
        fixed_bin_calibration(predictions, labels, bin_count=bin_count),
    )


def fixed_bin_calibration(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> tuple[CalibrationBin, ...]:
    """Compute deterministic equal-width calibration bins over ``[0, 1]``."""

    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("calibration predictions and targets must be matching rank-1 arrays")
    if bin_count < 1:
        raise ValueError("calibration bin_count must be positive")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("calibration values must be finite")
    if np.any(predictions < 0.0) or np.any(predictions > 1.0):
        raise ValueError("calibration predictions must be probabilities")
    indices = np.minimum((predictions * bin_count).astype(np.int64), bin_count - 1)
    bins: list[CalibrationBin] = []
    for index in range(bin_count):
        mask = indices == index
        count = int(np.count_nonzero(mask))
        bins.append(
            CalibrationBin(
                index / bin_count,
                (index + 1) / bin_count,
                count,
                None if count == 0 else float(np.mean(predictions[mask], dtype=np.float64)),
                None if count == 0 else float(np.mean(targets[mask], dtype=np.float64)),
            )
        )
    return tuple(bins)
