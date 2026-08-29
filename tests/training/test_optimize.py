"""Terminal-outcome optimizer uses sealed shards and reproduces fixed CPU runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from innovation_ai.training.checkpoint import load_checkpoint
from innovation_ai.training.compact_replay import sha256_digest
from innovation_ai.training.dataset import (
    DEFAULT_EXTRACTION_POLICY,
    DatasetCounts,
    DatasetManifest,
    DatasetShard,
    DatasetSourceShard,
    DatasetSplit,
    DatasetSplitMember,
    write_dataset_manifest,
)
from innovation_ai.training.encoding import build_encoder_manifest
from innovation_ai.training.optimize import (
    TrainingConfig,
    TrainingError,
    load_training_dataset,
    train_terminal_outcomes,
)


def _write_dataset(directory: Path) -> Path:
    encoder = build_encoder_manifest()
    train_ids = ("train-a", "train-b", "train-c", "train-d")
    validation_ids = ("validation-a", "validation-b")

    def arrays(ids: tuple[str, ...]) -> dict[str, np.ndarray]:
        features = np.zeros((len(ids), encoder.input_dimension), dtype=np.float32)
        features[:, 0] = np.asarray([-1.0, 1.0] * (len(ids) // 2), dtype=np.float32)
        targets = (features[:, 0] > 0.0).astype(np.float32)
        return {
            "features": features,
            "targets": targets,
            "episode_ids": np.asarray(ids),
            "viewers": np.zeros(len(ids), dtype=np.uint8),
            "action_kinds": np.zeros(len(ids), dtype=np.uint8),
            "decision_kinds": np.zeros(len(ids), dtype=np.uint8),
            "action_sequences": np.arange(1, len(ids) + 1, dtype=np.uint32),
        }

    directory.mkdir()
    shards: list[DatasetShard] = []
    for shard_id, split, ids in (
        ("train-00000", DatasetSplit.TRAIN, train_ids),
        ("validation-00000", DatasetSplit.VALIDATION, validation_ids),
    ):
        path = directory / f"{shard_id}.npz"
        np.savez(path, **cast(dict[str, Any], arrays(ids)))
        shards.append(
            DatasetShard(shard_id, split, ids, sha256_digest(path.read_bytes()), len(ids))
        )
    episodes = tuple(sorted(train_ids + validation_ids))
    memberships = tuple(
        DatasetSplitMember(
            episode_id,
            sha256_digest(f"setup:{episode_id}"),
            DatasetSplit.TRAIN if episode_id in train_ids else DatasetSplit.VALIDATION,
        )
        for episode_id in episodes
    )
    manifest = DatasetManifest(
        encoder_fingerprint=encoder.layout_fingerprint,
        encoder_version=encoder.encoder_version,
        extraction_policy=DEFAULT_EXTRACTION_POLICY,
        split_salt="test-salt",
        validation_fraction=0.2,
        source_shards=(DatasetSourceShard("source-00000", sha256_digest(b"source"), episodes),),
        split_membership=memberships,
        shards=tuple(shards),
        counts=DatasetCounts(6, 6, 4, 2, 4, 2),
    )
    manifest_path = directory / "manifest.json"
    write_dataset_manifest(manifest_path, manifest)
    return manifest_path


def _config() -> TrainingConfig:
    return TrainingConfig(
        seed=71,
        max_epochs=60,
        patience=60,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=1e-5,
        torch_num_threads=1,
    )


def _bundle_bytes(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _perf_counter_with_epoch_duration(epoch_duration: float) -> Callable[[], float]:
    calls = 0

    def perf_counter() -> float:
        nonlocal calls
        calls += 1
        return calls * epoch_duration

    return perf_counter


def test_training_config_has_strict_cpu_defaults() -> None:
    config = TrainingConfig()

    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-5
    assert config.batch_size == 1024
    assert config.seed == 0
    assert config.max_epochs == 100
    assert config.patience == 10
    assert config.torch_num_threads == 1
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(batch_size=0)


def test_tiny_cpu_terminal_training_overfits_and_saves_auditable_checkpoint(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path / "dataset")
    result = train_terminal_outcomes(manifest_path, tmp_path / "checkpoints", config=_config())
    loaded = load_checkpoint(result.checkpoint_directory, load_optimizer=True)

    assert result.report.train.bce < result.report.epoch_history[0].train_bce
    assert result.report.validation.brier < 0.05
    assert result.report.held_out_episode_count == 2
    assert result.report.held_out_game_count == 2
    assert len(result.report.validation.calibration) == 10
    assert all(record.examples_per_second > 0.0 for record in result.report.epoch_history)
    assert loaded.manifest.training_dataset_ids == (result.report.dataset_id,)
    assert loaded.manifest.parent_checkpoint_ids == ()
    assert loaded.manifest.generation == 0
    metrics_validation = cast(dict[str, object], loaded.metrics["validation"])
    assert metrics_validation["brier"] == result.report.validation.brier
    assert loaded.optimizer_state is not None


def test_training_is_reproducible_despite_volatile_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_dataset(tmp_path / "dataset")
    monkeypatch.setattr(
        "innovation_ai.training.optimize.time.perf_counter", _perf_counter_with_epoch_duration(1.0)
    )
    first = train_terminal_outcomes(manifest_path, tmp_path / "first", config=_config())
    monkeypatch.setattr(
        "innovation_ai.training.optimize.time.perf_counter", _perf_counter_with_epoch_duration(2.0)
    )
    second = train_terminal_outcomes(manifest_path, tmp_path / "second", config=_config())

    assert first.report.examples_per_second != second.report.examples_per_second
    assert first.report.payload()["examples_per_second"] == first.report.examples_per_second
    assert all(record.examples_per_second > 0.0 for record in first.report.epoch_history)
    assert first.checkpoint_directory.name == second.checkpoint_directory.name
    assert _bundle_bytes(first.checkpoint_directory) == _bundle_bytes(second.checkpoint_directory)

    checkpoint_metrics = json.loads((first.checkpoint_directory / "metrics.json").read_text())
    assert "examples_per_second" not in checkpoint_metrics
    assert all(
        "examples_per_second" not in record
        for record in cast(list[dict[str, object]], checkpoint_metrics["epoch_history"])
    )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first.report.best_epoch == second.report.best_epoch
    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key], rtol=0.0, atol=0.0)


def test_training_rejects_a_tampered_shard(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path / "dataset")
    train_terminal_outcomes(manifest_path, tmp_path / "checkpoints", config=_config())

    (manifest_path.parent / "train-00000.npz").write_bytes(b"tampered")
    with pytest.raises(TrainingError, match="digest"):
        load_training_dataset(manifest_path)
