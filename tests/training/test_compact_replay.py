"""Compact replay contracts stay independent from encoders and ML frameworks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from innovation_ai.innovation.actions import ChooseStartingMeldAction
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.types import CardId, PlayerId
from innovation_ai.training.compact_replay import (
    CompactEpisode,
    CompactReplayCompatibilityError,
    CompactReplayDivergenceError,
    CompactReplayProvenance,
    CompactReplayRecorder,
    CompactReplaySchemaError,
    CompactReplayShardError,
    CompactReplayShardManifest,
    DeterminizationProvenance,
    ExplorationProvenance,
    SeatPolicyProvenance,
    compact_episode_digest,
    dumps_compact_episode,
    loads_compact_episode,
    parse_canonical_json,
    read_compact_replay_shard,
    sha256_digest,
    verify_compact_episode,
    write_compact_replay_shard,
)


def _provenance() -> CompactReplayProvenance:
    return CompactReplayProvenance(
        producer_run_id="self-play-run-0001",
        resolved_config_digest=sha256_digest("resolved-config-v1"),
        generation=4,
        seat_mapping=(
            SeatPolicyProvenance(PlayerId.PLAYER_1, "policy-a", "checkpoint-a", "agent-rng-v1"),
            SeatPolicyProvenance(PlayerId.PLAYER_2, "policy-b", None, "agent-rng-v1"),
        ),
        exploration=ExplorationProvenance("softmax-v1", 0.25, "selector-rng-v1"),
        determinization=DeterminizationProvenance(
            "information-set-v1", "sampler-rng-v1", 1, "heuristic-v1", True
        ),
    )


def _episode(seed: int, episode_id: str) -> CompactEpisode:
    recorder = CompactReplayRecorder(build_setup_state(seed), episode_id, _provenance())
    for _ in range(200):
        decisions = recorder.decisions()
        if not decisions:
            break
        recorder.submit(decisions[0].legal_actions[0])
    return recorder.episode()


def test_compact_episode_round_trip_is_actions_only_and_verifiable() -> None:
    episode = _episode(1001, "episode-001")
    encoded = dumps_compact_episode(episode)

    assert '"decisions"' not in encoded
    assert '"observations"' not in encoded
    restored = loads_compact_episode(encoded)
    verified = verify_compact_episode(restored)

    assert restored == episode
    assert verified.transitions_replayed == episode.transition_count
    assert verified.state.terminal_result == episode.terminal_result
    assert compact_episode_digest(restored) == sha256_digest(encoded)
    assert episode.setup_digest.startswith("sha256:")


def test_compact_episode_rejects_noncanonical_edits_truncation_illegal_and_incompatible() -> None:
    episode = _episode(1002, "episode-002")

    with pytest.raises(CompactReplaySchemaError, match="canonical"):
        parse_canonical_json('{ "a":1}')
    with pytest.raises(CompactReplaySchemaError, match="duplicate"):
        parse_canonical_json('{"a":1,"a":1}')

    illegal = replace(
        episode,
        actions=(
            ChooseStartingMeldAction(episode.actions[0].decision_id, CardId("the-internet")),
            *episode.actions[1:],
        ),
    )
    with pytest.raises(CompactReplayDivergenceError, match="not legal"):
        verify_compact_episode(illegal)

    truncated = replace(
        episode,
        actions=episode.actions[:-1],
        transition_count=episode.transition_count - 1,
    )
    with pytest.raises(CompactReplayDivergenceError, match="truncated"):
        verify_compact_episode(truncated)

    with pytest.raises(CompactReplayCompatibilityError, match="incompatible engine"):
        verify_compact_episode(replace(episode, engine_version="999.0"))


def test_shard_is_preassigned_completion_order_independent_and_fixed_gzip(tmp_path: Path) -> None:
    first = _episode(1003, "episode-001")
    second = _episode(1004, "episode-002")
    manifest = CompactReplayShardManifest("shard-001", ("episode-001", "episode-002"))
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"

    first_digest = write_compact_replay_shard(first_path, manifest, [second, first])
    second_digest = write_compact_replay_shard(second_path, manifest, [first, second])

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes()[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert first_digest == second_digest == sha256_digest(first_path.read_bytes())
    assert read_compact_replay_shard(first_path, manifest, verify=True) == (first, second)

    wrong_manifest = CompactReplayShardManifest("shard-002", ("episode-001",))
    with pytest.raises(CompactReplayShardError, match="preassigned manifest"):
        read_compact_replay_shard(first_path, wrong_manifest)
