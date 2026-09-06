from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SAMPLED_MINIMAX_AGENT_DESCRIPTOR,
    SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
from innovation_ai.innovation.state import PUBLIC_COVERED_INFORMATION_POLICY_VERSION
from innovation_ai.search.contracts import (
    ROLLOUT_SEARCH_DESCRIPTOR_KIND,
    RolloutSearchDescriptor,
    SearchDescriptor,
)
from innovation_ai.training.checkpoint import (
    DEFAULT_ROLLOUT_FALLBACK_AGENT,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    PolicyDescriptor,
)
from innovation_ai.training.compact_replay import (
    CompactReplayShardManifest,
    read_compact_replay_shard,
)
from innovation_ai.training.self_play import (
    GenerationConfig,
    SeatPolicy,
    SelfPlayError,
    SelfPlayResumeError,
    default_learned_pool_seat_pairs,
    load_manifest,
    plan_generation,
    run_generation,
)


def _baseline_policies() -> tuple[SeatPolicy, SeatPolicy]:
    return (
        SeatPolicy(
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id,
            "baseline",
            descriptor=SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
        ),
        SeatPolicy(
            RANDOM_AGENT_DESCRIPTOR.descriptor_id,
            "baseline",
            descriptor=RANDOM_AGENT_DESCRIPTOR,
        ),
    )


def test_generation_plan_is_predeclared_immutable_and_round_trips(tmp_path: Path) -> None:
    heuristic, random = _baseline_policies()
    manifest = plan_generation(
        GenerationConfig("fixture-run", 91, 0, max_games_in_flight=2, shard_episode_limit=2),
        (random, heuristic),
        ((heuristic.policy_id, random.policy_id),),
        3,
    )
    assert tuple(a.episode_id for a in manifest.assignments) == (
        "episode-000000",
        "episode-000001",
        "episode-000002",
    )
    assert tuple(s.episode_ids for s in manifest.shards) == (
        ("episode-000000", "episode-000001"),
        ("episode-000002",),
    )
    run_generation(tmp_path, manifest, stop_requested=lambda: True)
    assert load_manifest(tmp_path / "run-manifest.json") == manifest
    assert not list((tmp_path / "replays").glob("*.gz"))

    incompatible = plan_generation(
        GenerationConfig("fixture-run", 92, 0),
        (random, heuristic),
        ((heuristic.policy_id, random.policy_id),),
        1,
    )
    with pytest.raises(SelfPlayResumeError, match="incompatible"):
        run_generation(tmp_path, incompatible, stop_requested=lambda: True)


def test_resume_rejects_an_incomplete_shard(tmp_path: Path) -> None:
    heuristic, random = _baseline_policies()
    manifest = plan_generation(
        GenerationConfig("fixture-run", 91, 0),
        (random, heuristic),
        ((heuristic.policy_id, random.policy_id),),
        1,
    )
    run_generation(tmp_path, manifest, stop_requested=lambda: True)
    replay = tmp_path / "replays"
    (replay / "shard-00000.jsonl.gz").write_bytes(b"partial")
    with pytest.raises(SelfPlayResumeError, match="incomplete or invalid"):
        run_generation(tmp_path, manifest)


def test_action_ceiling_stops_before_sealing_a_shard(tmp_path: Path) -> None:
    heuristic, random = _baseline_policies()
    manifest = plan_generation(
        GenerationConfig("ceiling-run", 91, 0, action_ceiling=1),
        (random, heuristic),
        ((heuristic.policy_id, random.policy_id),),
        1,
    )
    with pytest.raises(SelfPlayError, match="action ceiling"):
        run_generation(tmp_path, manifest)
    assert not list((tmp_path / "replays").glob("*.jsonl.gz"))
    failure = json.loads((tmp_path / "generation-failure.json").read_text())
    assert failure["format"] == "innovation-ai-self-play-action-ceiling-failure"
    assert failure["episode_id"] == "episode-000000"
    assert failure["setup_seed"] == manifest.assignments[0].setup_seed
    assert failure["action_count"] == 0
    assert failure["action_ceiling"] == 1
    assert failure["action_tail"] == []
    assert failure["current_state_hash"].startswith("sha256:")


@pytest.mark.slow
def test_bootstrap_generation_seals_and_resumes_without_duplicates(tmp_path: Path) -> None:
    heuristic, random = _baseline_policies()
    manifest = plan_generation(
        GenerationConfig("smoke-run", 1001, 0, max_games_in_flight=1, shard_episode_limit=1),
        (heuristic, random),
        ((heuristic.policy_id, random.policy_id),),
        1,
    )
    assert run_generation(tmp_path, manifest) == ("shard-00000",)
    shard = tmp_path / "replays" / "shard-00000.jsonl.gz"
    first = shard.read_bytes()
    assert run_generation(tmp_path, manifest) == ("shard-00000",)
    assert shard.read_bytes() == first


def test_search_baseline_generation_uses_supplied_descriptor_and_records_provenance(
    tmp_path: Path,
) -> None:
    search_descriptor = SearchDescriptor(
        root_turn_horizon=1,
        opponent_turn_horizon=1,
        starting_meld_horizon=1,
        determinization_count=1,
        route_transition_budget=1,
    )
    baseline_descriptor = AgentDescriptor(
        SAMPLED_MINIMAX_AGENT_DESCRIPTOR.name,
        SAMPLED_MINIMAX_AGENT_DESCRIPTOR.version,
        (("search_descriptor_id", search_descriptor.descriptor_id),),
    )
    search = SeatPolicy(
        baseline_descriptor.descriptor_id,
        "baseline",
        descriptor=baseline_descriptor,
    )
    random = _baseline_policies()[1]
    manifest = plan_generation(
        GenerationConfig(
            "search-smoke",
            1017,
            0,
            max_games_in_flight=1,
            shard_episode_limit=1,
            action_ceiling=2_000,
            validation_level="off",
        ),
        (search, random),
        ((search.policy_id, random.policy_id),),
        1,
    )

    assert run_generation(
        tmp_path,
        manifest,
        search_descriptors={search_descriptor.descriptor_id: search_descriptor},
    ) == ("shard-00000",)
    episode = read_compact_replay_shard(
        tmp_path / "replays" / "shard-00000.jsonl.gz",
        CompactReplayShardManifest("shard-00000", ("episode-000000",)),
        verify=True,
    )[0]

    assert episode.terminal_result is not None
    assert episode.information_policy_version == PUBLIC_COVERED_INFORMATION_POLICY_VERSION
    assert episode.provenance.seat_mapping[0].policy_descriptor_id == search.policy_id
    assert (
        episode.provenance.determinization.search_descriptor_id == search_descriptor.descriptor_id
    )


def test_rollout_baseline_resolves_generic_per_seat_algorithm_provenance(
    tmp_path: Path,
) -> None:
    rollout_descriptor = RolloutSearchDescriptor(determinization_count=1)
    baseline_descriptor = AgentDescriptor(
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.name,
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.version,
        (("search_descriptor_id", rollout_descriptor.descriptor_id),),
    )
    rollout = SeatPolicy(
        baseline_descriptor.descriptor_id,
        "baseline",
        descriptor=baseline_descriptor,
    )
    random = _baseline_policies()[1]
    manifest = plan_generation(
        GenerationConfig("rollout-provenance", 1019, 0),
        (rollout, random),
        ((rollout.policy_id, random.policy_id),),
        1,
    )

    import innovation_ai.training.self_play as self_play

    provenance = self_play._provenance(
        manifest,
        manifest.assignments[0],
        {rollout_descriptor.descriptor_id: rollout_descriptor},
    )

    rollout_seat = provenance.seat_mapping[0]
    assert rollout_seat.decision_algorithm_kind == ROLLOUT_SEARCH_DESCRIPTOR_KIND
    assert rollout_seat.decision_algorithm_descriptor_id == rollout_descriptor.descriptor_id
    assert rollout_seat.decision_algorithm_determinization_count == 1
    assert provenance.determinization.search_descriptor_id is None
    assert provenance.determinization.count == 0
    assert provenance.determinization.sampler_version == "per-seat-decision-algorithm-v2"

    confused = AgentDescriptor(
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.name,
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.version,
        (("search_descriptor_id", SearchDescriptor().descriptor_id),),
    )
    confused_policy = SeatPolicy(confused.descriptor_id, "baseline", descriptor=confused)
    confused_manifest = plan_generation(
        GenerationConfig("rollout-confused", 1020, 0),
        (confused_policy, random),
        ((confused_policy.policy_id, random.policy_id),),
        1,
    )
    with pytest.raises(SelfPlayError, match="decision-algorithm kind differs"):
        run_generation(tmp_path, confused_manifest, stop_requested=lambda: True)


def test_mixed_algorithm_families_and_counts_remain_per_seat_without_global_collapse() -> None:
    rollout_descriptor = RolloutSearchDescriptor(determinization_count=1)
    minimax_descriptor = SearchDescriptor(determinization_count=4)
    rollout_agent = AgentDescriptor(
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.name,
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.version,
        (("search_descriptor_id", rollout_descriptor.descriptor_id),),
    )
    minimax_agent = AgentDescriptor(
        SAMPLED_MINIMAX_AGENT_DESCRIPTOR.name,
        SAMPLED_MINIMAX_AGENT_DESCRIPTOR.version,
        (("search_descriptor_id", minimax_descriptor.descriptor_id),),
    )
    rollout = SeatPolicy(rollout_agent.descriptor_id, "baseline", descriptor=rollout_agent)
    minimax = SeatPolicy(minimax_agent.descriptor_id, "baseline", descriptor=minimax_agent)
    manifest = plan_generation(
        GenerationConfig("mixed-search-provenance", 1023, 0),
        (rollout, minimax),
        ((rollout.policy_id, minimax.policy_id),),
        1,
    )

    import innovation_ai.training.self_play as self_play

    provenance = self_play._provenance(
        manifest,
        manifest.assignments[0],
        {
            rollout_descriptor.descriptor_id: rollout_descriptor,
            minimax_descriptor.descriptor_id: minimax_descriptor,
        },
    )

    first, second = provenance.seat_mapping
    assert (first.decision_algorithm_kind, first.decision_algorithm_determinization_count) == (
        ROLLOUT_SEARCH_DESCRIPTOR_KIND,
        1,
    )
    assert (second.decision_algorithm_kind, second.decision_algorithm_determinization_count) == (
        "minimax-search",
        4,
    )
    assert first.decision_algorithm_descriptor_id == rollout_descriptor.descriptor_id
    assert second.decision_algorithm_descriptor_id == minimax_descriptor.descriptor_id
    assert provenance.determinization.search_descriptor_id is None
    assert provenance.determinization.count == 0


def test_v3_learned_rollout_resolves_and_validates_generic_provenance(tmp_path: Path) -> None:
    rollout_descriptor = RolloutSearchDescriptor(determinization_count=1)
    learned_descriptor = PolicyDescriptor(
        checkpoint_id="rollout-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
        fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
        fallback_algorithm_descriptor_id=rollout_descriptor.descriptor_id,
        hypothetical_continuation_policy=rollout_descriptor.continuation_agent_descriptor_id,
    )
    learned = SeatPolicy(
        learned_descriptor.policy_id,
        "learned",
        learned=learned_descriptor,
    )
    random = _baseline_policies()[1]
    manifest = plan_generation(
        GenerationConfig("learned-rollout", 1021, 0),
        (learned, random),
        ((learned.policy_id, random.policy_id),),
        1,
    )

    import innovation_ai.training.self_play as self_play

    provenance = self_play._provenance(
        manifest,
        manifest.assignments[0],
        {rollout_descriptor.descriptor_id: rollout_descriptor},
    )
    learned_seat = provenance.seat_mapping[0]
    assert learned_seat.decision_algorithm_kind == ROLLOUT_SEARCH_DESCRIPTOR_KIND
    assert learned_seat.decision_algorithm_descriptor_id == rollout_descriptor.descriptor_id
    assert learned_seat.decision_algorithm_determinization_count == 1
    assert provenance.determinization.search_descriptor_id is None
    assert provenance.determinization.count == 0

    mismatched = replace(
        learned_descriptor,
        hypothetical_continuation_policy="wrong-continuation-v1",
    )
    mismatched_seat = SeatPolicy(mismatched.policy_id, "learned", learned=mismatched)
    mismatched_manifest = plan_generation(
        GenerationConfig("learned-rollout-mismatch", 1022, 0),
        (mismatched_seat, random),
        ((mismatched_seat.policy_id, random.policy_id),),
        1,
    )
    with pytest.raises(SelfPlayError, match="hypothetical continuation differs"):
        run_generation(
            tmp_path,
            mismatched_manifest,
            checkpoint_root=tmp_path / "checkpoints",
            stop_requested=lambda: True,
            search_descriptors={rollout_descriptor.descriptor_id: rollout_descriptor},
        )


def test_v2_learned_generation_routes_search_fallback_with_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_descriptor = SearchDescriptor(
        root_turn_horizon=1,
        opponent_turn_horizon=1,
        starting_meld_horizon=1,
        determinization_count=1,
        route_transition_budget=1,
    )
    learned_descriptor = PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        search_descriptor_id=search_descriptor.descriptor_id,
        search_continuation_policy=DEFAULT_SEARCH_CONTINUATION_POLICY,
        fallback_algorithm_kind=None,
        fallback_algorithm_descriptor_id=None,
        hypothetical_continuation_policy=None,
        schema_version=MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )
    learned = SeatPolicy(
        learned_descriptor.policy_id,
        "learned",
        learned=learned_descriptor,
    )
    random = _baseline_policies()[1]
    evaluator_calls: list[int] = []

    class _Evaluator:
        def evaluate(self, positions: Any) -> tuple[float, ...]:
            evaluator_calls.append(len(positions))
            return (0.5,) * len(positions)

    class _EvaluatorCache:
        def __init__(self, checkpoint_root: object) -> None:
            del checkpoint_root

        def evaluator_for(self, descriptor: object) -> _Evaluator:
            del descriptor
            return _Evaluator()

    import innovation_ai.training.inference as inference

    monkeypatch.setattr(inference, "FrozenEvaluatorCache", _EvaluatorCache)
    manifest = plan_generation(
        GenerationConfig(
            "v2-search-smoke",
            1018,
            0,
            max_games_in_flight=1,
            shard_episode_limit=1,
            action_ceiling=2_000,
            validation_level="off",
        ),
        (learned, random),
        ((learned.policy_id, random.policy_id),),
        1,
    )

    assert run_generation(
        tmp_path,
        manifest,
        checkpoint_root=tmp_path / "unused-checkpoints",
        search_descriptors={search_descriptor.descriptor_id: search_descriptor},
    ) == ("shard-00000",)
    episode = read_compact_replay_shard(
        tmp_path / "replays" / "shard-00000.jsonl.gz",
        CompactReplayShardManifest("shard-00000", ("episode-000000",)),
        verify=True,
    )[0]

    assert evaluator_calls
    assert episode.provenance.seat_mapping[0].policy_descriptor_id == learned.policy_id
    assert episode.provenance.seat_mapping[0].checkpoint_id == "fixture-checkpoint"
    assert (
        episode.provenance.determinization.search_descriptor_id == search_descriptor.descriptor_id
    )


def test_default_learned_pool_mix_is_fixed_50_25_25_with_balanced_seats() -> None:
    pairs = default_learned_pool_seat_pairs("latest", "previous", ("older-b", "older-a"))
    opponents = tuple(second if first == "latest" else first for first, second in pairs)

    assert opponents.count("latest") == 4
    assert opponents.count("previous") == 2
    assert opponents.count("older-a") == 1
    assert opponents.count("older-b") == 1
    assert sum(first == "latest" for first, _ in pairs) == sum(
        second == "latest" for _, second in pairs
    )
