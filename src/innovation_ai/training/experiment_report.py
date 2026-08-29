"""Canonical, read-only summaries for Milestone 3 training experiments.

The builder deliberately derives every value from existing replay, dataset, checkpoint, policy, and
arena artifacts.  It never rewrites a manifest, shard, or checkpoint, making the summary suitable
for a CLI command without creating a competing source of truth.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from innovation_ai.training.checkpoint import load_checkpoint_manifest, load_policy_descriptor
from innovation_ai.training.compact_replay import (
    CompactEpisode,
    CompactReplayShardManifest,
    JsonValue,
    canonical_json,
    read_compact_replay_shard,
    sha256_digest,
)
from innovation_ai.training.dataset import DatasetSplit, read_dataset_manifest
from innovation_ai.training.optimize import dataset_id
from innovation_ai.training.self_play import load_manifest

EXPERIMENT_REPORT_FORMAT = "innovation-ai-training-experiment-report"
EXPERIMENT_REPORT_SCHEMA_VERSION = 1


class ExperimentReportError(ValueError):
    """An experiment artifact is missing, malformed, or mutually inconsistent."""


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Versioned canonical experiment payload with deterministic Markdown rendering."""

    payload: dict[str, object]

    def dumps(self) -> str:
        """Return compact canonical JSON without a trailing newline."""

        return canonical_json(cast(JsonValue, self.payload))

    def to_markdown(self) -> str:
        """Render the concise, stable human-facing report."""

        return render_experiment_report(self)


def build_experiment_report(
    run_dir: str | Path,
    *,
    arena_report_path: str | Path | None = None,
    failure_counters: Mapping[str, int] | None = None,
) -> ExperimentReport:
    """Build a read-only report from an iteration directory and optional arena JSON.

    ``run_dir`` contains generation directories such as ``bootstrap`` and ``learned``.  Each
    discovered ``run-manifest.json`` must have sealed replay shards; a sibling ``dataset``
    directory and checkpoint bundles are summarized when present.  Missing optional artifacts are
    represented by empty lists or ``null`` rather than fabricated measurements.
    """

    root = Path(run_dir)
    if not root.is_dir():
        raise ExperimentReportError(f"run directory does not exist: {root}")
    failures = _failure_counters(failure_counters)
    generation_dirs = sorted({path.parent for path in root.rglob("run-manifest.json")})
    if not generation_dirs:
        raise ExperimentReportError(f"no generation manifests beneath {root}")

    generations = [_generation_payload(directory) for directory in generation_dirs]
    root_telemetry = _combined_telemetry(root)
    resolved_config = _read_optional_json_object(root / "resolved-config.json", "resolved config")
    iteration_state = _read_optional_json_object(root / "iteration-state.json", "iteration state")
    telemetry_by_generation = {
        cast(int, generation["generation"]): cast(dict[str, JsonValue], generation["telemetry"])
        for generation in generations
    }
    checkpoints = _checkpoint_payloads(root, telemetry_by_generation, root_telemetry)
    policies = _policy_payloads(root)
    policy_ids_by_checkpoint: dict[str, list[str]] = {}
    for policy in policies:
        policy_ids_by_checkpoint.setdefault(cast(str, policy["checkpoint_id"]), []).append(
            cast(str, policy["policy_id"])
        )
    for checkpoint in checkpoints:
        checkpoint["policy_ids"] = policy_ids_by_checkpoint.get(
            cast(str, checkpoint["checkpoint_id"]), []
        )
    arena = _read_arena(Path(arena_report_path)) if arena_report_path is not None else None
    payload: dict[str, object] = {
        "format": EXPERIMENT_REPORT_FORMAT,
        "schema_version": EXPERIMENT_REPORT_SCHEMA_VERSION,
        "run_id": root.name,
        "resolved_config": resolved_config,
        "resolved_config_digest": (
            resolved_config.get("config_digest") if resolved_config is not None else None
        ),
        "generations": generations,
        "checkpoints": checkpoints,
        "policies": policies,
        "failure_counters": failures,
        "telemetry": root_telemetry,
        "arena": arena,
    }
    if iteration_state is not None:
        for key in ("bootstrap_policy_id", "candidate_policy_id", "candidate_checkpoint_id"):
            value = iteration_state.get(key)
            if value is not None and not isinstance(value, str):
                raise ExperimentReportError(f"iteration state {key} must be a string")
            payload[key] = value
    return ExperimentReport(payload)


def write_experiment_report(
    run_dir: str | Path,
    *,
    arena_report_path: str | Path | None = None,
    failure_counters: Mapping[str, int] | None = None,
    json_name: str = "experiment-report.json",
    markdown_name: str = "experiment-report.md",
) -> ExperimentReport:
    """Build and atomically write canonical JSON plus concise Markdown under ``run_dir``."""

    root = Path(run_dir)
    report = build_experiment_report(
        root, arena_report_path=arena_report_path, failure_counters=failure_counters
    )
    _atomic_write(root / json_name, (report.dumps() + "\n").encode("ascii"))
    _atomic_write(root / markdown_name, report.to_markdown().encode("utf-8"))
    return report


def render_experiment_report(report: ExperimentReport) -> str:
    """Render report JSON as a compact Markdown status table."""

    payload = report.payload
    lines = [
        f"# Experiment {payload['run_id']}",
        "",
        "| generation | episodes | transitions | examples (train/validation) | "
        "target mean (train/validation) |",
        "|---|---:|---:|---:|---:|",
    ]
    for generation in cast(list[dict[str, object]], payload["generations"]):
        counts = cast(dict[str, int], generation["counts"])
        targets = cast(dict[str, dict[str, object]], generation["targets"])
        lines.append(
            "| {generation} | {episodes} | {transitions} | {examples} ({train}/{validation}) | "
            "{train_mean}/{validation_mean} |".format(
                generation=generation["generation"],
                episodes=counts["episodes"],
                transitions=counts["transitions"],
                examples=counts["examples"],
                train=counts["train_examples"],
                validation=counts["validation_examples"],
                train_mean=_number(targets["train"]["mean"]),
                validation_mean=_number(targets["validation"]["mean"]),
            )
        )
    lines += [
        "",
        "## Checkpoints",
        "",
        "| generation | checkpoint | best epoch | validation BCE / Brier | examples/s |",
        "|---:|---|---:|---:|---:|",
    ]
    for checkpoint in cast(list[dict[str, object]], payload["checkpoints"]):
        training = cast(dict[str, object], checkpoint["training"])
        throughput = cast(dict[str, object], checkpoint["throughput"])
        validation = cast(dict[str, object], training.get("validation", {}))
        lines.append(
            f"| {checkpoint['generation']} | {checkpoint['checkpoint_id']} | "
            f"{training.get('best_epoch', '—')} | {_number(validation.get('bce'))} / "
            f"{_number(validation.get('brier'))} | "
            f"{_number(throughput.get('examples_per_second'))} |"
        )
    failures = cast(dict[str, int], payload["failure_counters"])
    lines += [
        "",
        "## Failure counters",
        "",
        ", ".join(f"{key}: {value}" for key, value in failures.items()),
    ]
    arena = payload["arena"]
    if arena is not None:
        arena_data = cast(dict[str, object], arena)
        weighted = cast(dict[str, object], arena_data.get("weighted_pool", {}))
        lines += [
            "",
            "## Arena",
            "",
            f"candidate: {arena_data.get('candidate_policy_id', '—')}; "
            f"weighted utility: {_number(weighted.get('mean_pair_utility'))}",
        ]
    return "\n".join(lines) + "\n"


def _generation_payload(directory: Path) -> dict[str, object]:
    manifest = load_manifest(directory / "run-manifest.json")
    replay_dir = directory / "replays"
    episodes: list[CompactEpisode] = []
    for shard in manifest.shards:
        path = replay_dir / f"{shard.shard_id}.jsonl.gz"
        if not path.is_file():
            raise ExperimentReportError(f"missing sealed replay shard: {path}")
        try:
            episodes.extend(
                read_compact_replay_shard(
                    path,
                    CompactReplayShardManifest(shard.shard_id, shard.episode_ids),
                    verify=True,
                )
            )
        except Exception as error:
            raise ExperimentReportError(f"could not read replay shard {path}: {error}") from error
    resolved = {episode.provenance.resolved_config_digest for episode in episodes}
    if len(resolved) != 1:
        raise ExperimentReportError("generation replays have inconsistent resolved config digests")
    if {episode.provenance.generation for episode in episodes} != {manifest.config.generation}:
        raise ExperimentReportError("generation replay provenance differs from manifest")
    dataset = _dataset_payload(directory / "dataset")
    dataset_counts = None if dataset is None else cast(dict[str, int], dataset["counts"])
    counts = {
        "episodes": len(episodes),
        "transitions": sum(episode.transition_count for episode in episodes),
        "examples": 0 if dataset_counts is None else dataset_counts["examples"],
        "train_episodes": 0 if dataset_counts is None else dataset_counts["train_episodes"],
        "validation_episodes": (
            0 if dataset_counts is None else dataset_counts["validation_episodes"]
        ),
        "train_examples": 0 if dataset_counts is None else dataset_counts["train_examples"],
        "validation_examples": (
            0 if dataset_counts is None else dataset_counts["validation_examples"]
        ),
    }
    generation_telemetry = _read_telemetry(directory / "generation-telemetry.json")
    training_telemetry = _read_telemetry(directory / "training-telemetry.json")
    telemetry = {**generation_telemetry, **training_telemetry}
    return {
        "generation": manifest.config.generation,
        "name": directory.name,
        "resolved_config": cast(JsonValue, manifest.payload()),
        "resolved_config_digest": next(iter(resolved)),
        "counts": counts,
        "dataset": dataset,
        "targets": {"train": _target_empty(), "validation": _target_empty()}
        if dataset is None
        else dataset["targets"],
        "telemetry": telemetry,
        "generation_telemetry": generation_telemetry,
        "training_telemetry": training_telemetry,
    }


def _dataset_payload(directory: Path) -> dict[str, object] | None:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = read_dataset_manifest(path)
    except Exception as error:
        raise ExperimentReportError(f"could not read dataset manifest {path}: {error}") from error
    target_values: dict[DatasetSplit, list[np.ndarray]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
    }
    for shard in manifest.shards:
        shard_path = directory / f"{shard.shard_id}.npz"
        try:
            raw_targets = shard_path.read_bytes()
            if sha256_digest(raw_targets) != shard.sha256:
                raise ExperimentReportError(
                    f"dataset shard digest differs from manifest for {shard.shard_id}"
                )
            with np.load(shard_path, allow_pickle=False) as data:
                targets = np.asarray(data["targets"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as error:
            raise ExperimentReportError(
                f"could not read dataset targets {shard_path}: {error}"
            ) from error
        if targets.ndim != 1 or targets.shape[0] != shard.example_count:
            raise ExperimentReportError(
                f"dataset target count differs from manifest for {shard.shard_id}"
            )
        target_values[shard.split].append(targets)
    target_summaries = {split: _target_summary(target_values[split]) for split in DatasetSplit}
    train_values = _joined_targets(target_values[DatasetSplit.TRAIN])
    validation_values = _joined_targets(target_values[DatasetSplit.VALIDATION])
    if len(train_values) and len(validation_values):
        train_mean = float(train_values.mean())
        target_summaries[DatasetSplit.VALIDATION]["constant_mean_brier"] = float(
            np.mean((validation_values - train_mean) ** 2)
        )
    return {
        "dataset_id": dataset_id(manifest),
        "manifest_sha256": sha256_digest(path.read_bytes()),
        "counts": {
            "episodes": manifest.counts.episode_count,
            "examples": manifest.counts.example_count,
            "train_episodes": manifest.counts.train_episode_count,
            "validation_episodes": manifest.counts.validation_episode_count,
            "train_examples": manifest.counts.train_example_count,
            "validation_examples": manifest.counts.validation_example_count,
        },
        "targets": {
            "train": target_summaries[DatasetSplit.TRAIN],
            "validation": target_summaries[DatasetSplit.VALIDATION],
        },
    }


def _target_empty() -> dict[str, object]:
    return {"count": 0, "mean": None, "constant_mean_brier": None}


def _joined_targets(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _target_summary(parts: list[np.ndarray]) -> dict[str, object]:
    values = _joined_targets(parts)
    if not len(values):
        return _target_empty()
    mean = float(values.mean())
    return {
        "count": len(values),
        "mean": mean,
        "constant_mean_brier": float(np.mean((values - mean) ** 2)),
    }


def _checkpoint_payloads(
    root: Path,
    telemetry_by_generation: Mapping[int, Mapping[str, JsonValue]],
    root_telemetry: Mapping[str, JsonValue],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(root.glob("checkpoints/*/manifest.json")):
        try:
            manifest = load_checkpoint_manifest(path.parent)
            metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ExperimentReportError(
                f"could not read checkpoint {path.parent}: {error}"
            ) from error
        if not isinstance(metrics, dict):
            raise ExperimentReportError(f"checkpoint metrics must be an object: {path.parent}")
        result.append(
            {
                "checkpoint_id": manifest.checkpoint_id,
                "generation": manifest.generation,
                "policy_ids": [],
                "parent_checkpoint_ids": list(manifest.parent_checkpoint_ids),
                "dataset_ids": list(manifest.training_dataset_ids),
                "training": cast(JsonValue, metrics),
                "throughput": _throughput(
                    {
                        **root_telemetry,
                        **(telemetry_by_generation.get(manifest.generation) or {}),
                    }
                ),
            }
        )
    return result


def _policy_payloads(root: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("policies/*.json")):
        try:
            policy = load_policy_descriptor(path)
        except ValueError as error:
            raise ExperimentReportError(
                f"could not read policy descriptor {path}: {error}"
            ) from error
        result.append({"policy_id": policy.policy_id, "checkpoint_id": policy.checkpoint_id})
    return result


def _read_arena(path: Path) -> JsonValue:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentReportError(f"could not read arena report {path}: {error}") from error
    try:
        canonical_json(cast(JsonValue, value))
    except ValueError as error:
        raise ExperimentReportError(f"arena report is not finite JSON: {error}") from error
    if not isinstance(value, dict) or value.get("format") != "innovation-ai-arena-report":
        raise ExperimentReportError("arena report is not an innovation-ai arena report")
    return cast(JsonValue, value)


def _read_optional_json_object(path: Path, label: str) -> dict[str, JsonValue] | None:
    """Read one optional finite JSON object sidecar."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        canonical_json(cast(JsonValue, value))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ExperimentReportError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExperimentReportError(f"{label} must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _combined_telemetry(directory: Path) -> dict[str, JsonValue]:
    """Merge generation and training sidecars; training wins duplicate keys."""

    return {
        **_read_telemetry(directory / "generation-telemetry.json"),
        **_read_telemetry(directory / "training-telemetry.json"),
    }


def _read_telemetry(path: Path) -> dict[str, JsonValue]:
    """Read optional finite JSON stage telemetry without making it a new source of truth."""

    value = _read_optional_json_object(path, "stage telemetry")
    return {} if value is None else value


def _throughput(telemetry: Mapping[str, JsonValue]) -> dict[str, float | None]:
    """Extract optional measured throughput from a stage telemetry payload."""

    nested = telemetry.get("throughput")
    if nested is None:
        source: Mapping[str, JsonValue] = telemetry
    elif isinstance(nested, dict):
        source = {**nested, **telemetry}
    else:
        raise ExperimentReportError("stage telemetry throughput must be an object")
    return {
        "examples_per_second": _telemetry_rate(source.get("examples_per_second")),
        "actions_per_second": _telemetry_rate(source.get("actions_per_second")),
    }


def _telemetry_rate(value: JsonValue | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentReportError("stage telemetry throughput rates must be numbers or null")
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0:
        raise ExperimentReportError(
            "stage telemetry throughput rates must be finite and non-negative"
        )
    return rate


def _failure_counters(values: Mapping[str, int] | None) -> dict[str, int]:
    counters = {
        "sampler_failures": 0,
        "replay_failures": 0,
        "integrity_failures": 0,
        "action_ceiling_failures": 0,
    }
    for key, value in (values or {}).items():
        if (
            key not in counters
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ExperimentReportError(f"invalid failure counter {key!r}")
        counters[key] = value
    return counters


def _number(value: object) -> str:
    return "—" if value is None else f"{cast(float, value):.4f}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
