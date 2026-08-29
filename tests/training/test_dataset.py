"""Replay-derived dataset examples remain verified, player-safe, and deterministic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from innovation_ai.innovation.actions import DogmaAction
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.compact_replay import (
    CompactEpisode,
    CompactReplayDivergenceError,
    CompactReplayProvenance,
    CompactReplayRecorder,
    CompactReplayShardManifest,
    DeterminizationProvenance,
    ExplorationProvenance,
    SeatPolicyProvenance,
    setup_provenance_digest,
    sha256_digest,
    write_compact_replay_shard,
)
from innovation_ai.training.dataset import (
    DatasetMaterializationError,
    extract_value_position_examples,
    load_dataset_shard,
    materialize_dataset,
)
from innovation_ai.training.encoding import FlatObservationEncoder


def _provenance() -> CompactReplayProvenance:
    return CompactReplayProvenance(
        producer_run_id="dataset-test-run",
        resolved_config_digest=sha256_digest("dataset-test-config"),
        generation=0,
        seat_mapping=(
            SeatPolicyProvenance(PlayerId.PLAYER_1, "test-policy", None, "rng-v1"),
            SeatPolicyProvenance(PlayerId.PLAYER_2, "test-policy", None, "rng-v1"),
        ),
        exploration=ExplorationProvenance("first-legal-v1", 0.0, "rng-v1"),
        determinization=DeterminizationProvenance("none-v1", "rng-v1", 0, None, True),
    )


def _episode(seed: int, episode_id: str, *, prefer_dogma: bool = False) -> CompactEpisode:
    recorder = CompactReplayRecorder(build_setup_state(seed), episode_id, _provenance())
    for _ in range(400):
        decisions = recorder.decisions()
        if not decisions:
            break
        decision = decisions[0]
        action = next(
            (
                candidate
                for candidate in decision.legal_actions
                if prefer_dogma and isinstance(candidate, DogmaAction)
            ),
            decision.legal_actions[0],
        )
        recorder.submit(action)
    return recorder.episode()


def _source(path: Path, episodes: list[CompactEpisode]) -> Path:
    manifest = CompactReplayShardManifest(
        "source-001", tuple(sorted(episode.episode_id for episode in episodes))
    )
    write_compact_replay_shard(path, manifest, list(reversed(episodes)))
    return path


def test_extraction_replays_verified_actions_and_matches_online_encoding() -> None:
    episode = _episode(1, "effect-episode", prefer_dogma=True)

    examples = extract_value_position_examples(episode)

    assert examples
    assert all(example.decision_kind.value != "starting-meld" for example in examples)
    assert {example.decision_kind.value for example in examples} == {
        "turn-action",
        "effect-choice",
    }
    assert all(example.position.observation.viewer is example.viewer for example in examples)
    assert all(example.target in (0.0, 0.5, 1.0) for example in examples)

    encoder = FlatObservationEncoder()
    offline = encoder.encode_batch(tuple(example.position for example in examples))
    online = np.stack([encoder.encode(example.position) for example in examples])
    np.testing.assert_array_equal(offline, online)


def test_extraction_rejects_edited_replays_before_emitting_examples() -> None:
    episode = _episode(7, "edited-episode")
    edited = replace(episode, final_state_hash=sha256_digest("not-the-final-state"))

    with pytest.raises(CompactReplayDivergenceError, match="final state hash"):
        extract_value_position_examples(edited)


def test_materialization_groups_duplicate_setup_provenance_and_is_resumable_deterministic(
    tmp_path: Path,
) -> None:
    first = _episode(22, "duplicate-a")
    duplicate = _episode(22, "duplicate-b")
    other = _episode(23, "other")
    source = _source(tmp_path / "source-001.jsonl.gz", [first, duplicate, other])

    one = materialize_dataset(
        [source], tmp_path / "dataset-one", validation_fraction=0.5, episodes_per_shard=1
    )
    two = materialize_dataset(
        [source], tmp_path / "dataset-two", validation_fraction=0.5, episodes_per_shard=1
    )

    memberships = {member.episode_id: member for member in one.split_membership}
    assert memberships["duplicate-a"].setup_provenance_digest == setup_provenance_digest(
        first.setup
    )
    assert memberships["duplicate-a"].split is memberships["duplicate-b"].split
    assert one.counts == two.counts
    assert one.source_shards[0].sha256 == sha256_digest(source.read_bytes())
    assert (tmp_path / "dataset-one" / "manifest.json").read_bytes() == (
        tmp_path / "dataset-two" / "manifest.json"
    ).read_bytes()
    for shard in one.shards:
        assert (tmp_path / "dataset-one" / f"{shard.shard_id}.npz").read_bytes() == (
            tmp_path / "dataset-two" / f"{shard.shard_id}.npz"
        ).read_bytes()
        arrays = load_dataset_shard(tmp_path / "dataset-one" / f"{shard.shard_id}.npz")
        assert arrays["features"].dtype == np.float32
        assert arrays["targets"].dtype == np.float32
        assert arrays["episode_ids"].dtype.kind == "U"
        assert arrays["features"].shape[0] == shard.example_count

    # A completed deterministic output is a no-op; a stale shard is never silently overwritten.
    assert (
        materialize_dataset(
            [source], tmp_path / "dataset-one", validation_fraction=0.5, episodes_per_shard=1
        )
        == one
    )
    stale = tmp_path / "dataset-one" / f"{one.shards[0].shard_id}.npz"
    stale.write_bytes(b"stale")
    with pytest.raises(DatasetMaterializationError, match="differs"):
        materialize_dataset(
            [source], tmp_path / "dataset-one", validation_fraction=0.5, episodes_per_shard=1
        )


def test_terminal_utility_labels_are_viewer_relative() -> None:
    episode = _episode(1, "winner-episode", prefer_dogma=True)
    assert episode.terminal_result.winners == (PlayerId.PLAYER_2,)

    examples = extract_value_position_examples(episode)

    assert {example.target for example in examples if example.viewer is PlayerId.PLAYER_1} == {0.0}
    assert {example.target for example in examples if example.viewer is PlayerId.PLAYER_2} == {1.0}
