from __future__ import annotations

from pathlib import Path

import pytest

from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
)
from innovation_ai.training.self_play import (
    GenerationConfig,
    SeatPolicy,
    SelfPlayResumeError,
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
