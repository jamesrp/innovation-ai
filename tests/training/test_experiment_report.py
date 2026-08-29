from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from innovation_ai.agents.descriptors import SIMPLE_HEURISTIC_AGENT_DESCRIPTOR
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.checkpoint import (
    PolicyDescriptor,
    load_checkpoint_manifest,
    save_checkpoint,
)
from innovation_ai.training.compact_replay import (
    CompactReplayProvenance,
    CompactReplayRecorder,
    CompactReplayShardManifest,
    DeterminizationProvenance,
    ExplorationProvenance,
    SeatPolicyProvenance,
    sha256_digest,
    write_compact_replay_shard,
)
from innovation_ai.training.dataset import DatasetSplit, materialize_dataset, read_dataset_manifest
from innovation_ai.training.encoding import build_encoder_manifest
from innovation_ai.training.experiment_report import (
    EXPERIMENT_REPORT_FORMAT,
    ExperimentReportError,
    build_experiment_report,
    write_experiment_report,
)
from innovation_ai.training.model import ValueNetwork
from innovation_ai.training.self_play import (
    GenerationConfig,
    SeatPolicy,
    plan_generation,
    save_manifest,
)


def test_builder_derives_canonical_counts_targets_metrics_and_ids(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    generation_dir = root / "bootstrap"
    baseline = SeatPolicy(
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id,
        "baseline",
        descriptor=SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    )
    manifest = plan_generation(
        GenerationConfig("pilot-bootstrap", 19, 0, max_games_in_flight=1, shard_episode_limit=8),
        (baseline,),
        ((baseline.policy_id, baseline.policy_id),),
        8,
    )
    save_manifest(generation_dir / "run-manifest.json", manifest)
    episodes = []
    provenance_digest = sha256_digest(
        json.dumps(manifest.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )
    for setup_seed, assignment in enumerate(manifest.assignments, start=1):
        provenance = CompactReplayProvenance(
            manifest.config.run_id,
            provenance_digest,
            manifest.config.generation,
            (
                SeatPolicyProvenance(
                    PlayerId.PLAYER_1,
                    assignment.seat_policies[0].policy_id,
                    None,
                    "sha256-domain-separated-v1",
                ),
                SeatPolicyProvenance(
                    PlayerId.PLAYER_2,
                    assignment.seat_policies[1].policy_id,
                    None,
                    "sha256-domain-separated-v1",
                ),
            ),
            ExplorationProvenance("temperature-softmax-v1", 0.0, "sha256-domain-separated-v1"),
            DeterminizationProvenance(
                "information-set-sampler-v1", "sha256-counter-v1", 0, "simple-heuristic", True
            ),
        )
        recorder = CompactReplayRecorder(
            build_setup_state(setup_seed), assignment.episode_id, provenance
        )
        for _ in range(400):
            decisions = recorder.decisions()
            if not decisions:
                break
            recorder.submit(decisions[0].legal_actions[0])
        episodes.append(recorder.episode())
    shard = manifest.shards[0]
    write_compact_replay_shard(
        generation_dir / "replays" / f"{shard.shard_id}.jsonl.gz",
        CompactReplayShardManifest(shard.shard_id, shard.episode_ids),
        episodes,
    )
    materialize_dataset(
        sorted((generation_dir / "replays").glob("*.jsonl.gz")),
        generation_dir / "dataset",
        validation_fraction=0.5,
        split_salt="report-test-salt",
    )

    encoder = build_encoder_manifest()
    checkpoint_dir = save_checkpoint(
        root / "checkpoints",
        ValueNetwork(encoder.input_dimension),
        encoder,
        metrics={
            "best_epoch": 2,
            "epochs_completed": 3,
            "train": {"bce": 0.6, "brier": 0.2, "mean_prediction": 0.5, "calibration": []},
            "validation": {
                "bce": 0.7,
                "brier": 0.25,
                "mean_prediction": 0.5,
                "calibration": [],
            },
        },
        training_dataset_ids=("dataset-test",),
        generation=0,
    )
    checkpoint = load_checkpoint_manifest(checkpoint_dir)
    policy = PolicyDescriptor.from_checkpoint(checkpoint, temperature=0.15, determinization_count=1)
    (root / "policies").mkdir(parents=True)
    policy.save(root / "policies" / f"{policy.policy_id}.json")

    (root / "resolved-config.json").write_text(
        '{"config_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","seed":19}\n',
        encoding="utf-8",
    )
    (root / "iteration-state.json").write_text(
        '{"bootstrap_policy_id":"bootstrap-policy","candidate_checkpoint_id":"candidate-checkpoint","candidate_policy_id":"candidate-policy","config_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","format":"innovation-ai-iteration-state","schema_version":1}\n',
        encoding="utf-8",
    )
    (generation_dir / "generation-telemetry.json").write_text(
        '{"actions_per_second":8.5}\n', encoding="utf-8"
    )
    (root / "training-telemetry.json").write_text(
        '{"throughput":{"actions_per_second":8.5,"examples_per_second":12.5}}\n',
        encoding="utf-8",
    )
    report = build_experiment_report(root)
    payload = cast(dict[str, Any], report.payload)
    generation = payload["generations"][0]
    assert payload["format"] == EXPERIMENT_REPORT_FORMAT
    assert payload["resolved_config"]["seed"] == 19
    assert payload["resolved_config_digest"] == (
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert payload["bootstrap_policy_id"] == "bootstrap-policy"
    assert payload["candidate_policy_id"] == "candidate-policy"
    assert payload["candidate_checkpoint_id"] == "candidate-checkpoint"
    assert generation["generation_telemetry"] == {"actions_per_second": 8.5}
    assert generation["training_telemetry"] == {}
    assert generation["counts"]["episodes"] == 8
    assert generation["counts"]["transitions"] > 0
    target_count = (
        generation["targets"]["train"]["count"] + generation["targets"]["validation"]["count"]
    )
    assert target_count == generation["counts"]["examples"]
    assert generation["targets"]["validation"]["constant_mean_brier"] is not None
    dataset_manifest = read_dataset_manifest(generation_dir / "dataset" / "manifest.json")
    split_targets: dict[DatasetSplit, list[np.ndarray]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
    }
    for dataset_shard in dataset_manifest.shards:
        with np.load(generation_dir / "dataset" / f"{dataset_shard.shard_id}.npz") as arrays:
            split_targets[dataset_shard.split].append(arrays["targets"])
    train_targets = np.concatenate(split_targets[DatasetSplit.TRAIN])
    validation_targets = np.concatenate(split_targets[DatasetSplit.VALIDATION])
    expected_brier = float(np.mean((validation_targets - train_targets.mean()) ** 2))
    assert generation["targets"]["validation"]["constant_mean_brier"] == expected_brier
    assert payload["checkpoints"][0]["checkpoint_id"] == checkpoint.checkpoint_id
    assert payload["checkpoints"][0]["throughput"] == {
        "examples_per_second": 12.5,
        "actions_per_second": 8.5,
    }
    assert payload["failure_counters"]["sampler_failures"] == 0
    assert report.dumps() == build_experiment_report(root).dumps()

    arena_path = root / "arena-report.json"
    arena_path.write_text(
        '{"candidate_policy_id":"arena-candidate","format":"innovation-ai-arena-report",'
        '"weighted_pool":{"mean_pair_utility":0.625}}\n',
        encoding="utf-8",
    )
    arena_report = build_experiment_report(
        root,
        arena_report_path=arena_path,
        failure_counters={"sampler_failures": 1},
    )
    arena_payload = cast(dict[str, Any], arena_report.payload)
    assert arena_payload["failure_counters"]["sampler_failures"] == 1
    assert "weighted utility: 0.6250" in arena_report.to_markdown()

    written = write_experiment_report(root)
    assert (root / "experiment-report.json").read_text(encoding="ascii") == written.dumps() + "\n"
    assert "# Experiment pilot" in (root / "experiment-report.md").read_text(encoding="utf-8")


def test_builder_rejects_missing_runs_and_invalid_counters(tmp_path: Path) -> None:
    with pytest.raises(ExperimentReportError, match="does not exist"):
        build_experiment_report(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExperimentReportError, match="no generation manifests"):
        build_experiment_report(empty)

    with pytest.raises(ExperimentReportError, match="invalid failure counter"):
        build_experiment_report(empty, failure_counters={"sampler_failures": -1})
