from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from innovation_ai.harness.arena import (
    ArenaGameResult,
    ArenaManifest,
    ArenaReport,
    ArenaResult,
    BootstrapConfig,
    PolicyDescriptor,
    PolicyPool,
    PoolEntry,
    build_arena_report,
    dumps_arena_result,
    plan_match_pair,
)
from innovation_ai.harness.arena_runner import (
    PROMOTION_PAIR_COUNT,
    ArenaExecutionError,
    ArenaRunner,
    ChampionManifest,
    PromotionDecision,
    dumps_champion_manifest,
    loads_champion_manifest,
    promotion_outcome,
    write_champion,
)
from innovation_ai.harness.engine import InnovationEngineAdapter
from innovation_ai.innovation.state import TerminalReason, TerminalResult
from innovation_ai.innovation.types import PlayerId


def _baseline_manifest() -> ArenaManifest:
    return ArenaManifest(
        "arena-execution",
        "heuristic",
        PolicyPool("explicit-baselines", (PoolEntry("random", 1),)),
        (plan_match_pair("seed-808", 808, "heuristic", "random"),),
        BootstrapConfig(seed=7, resamples=3),
    )


def test_executor_runs_exact_swapped_games_deterministically() -> None:
    manifest = _baseline_manifest()
    runner = ArenaRunner(
        InnovationEngineAdapter(),
        {
            "heuristic": PolicyDescriptor("heuristic", "heuristic"),
            "random": PolicyDescriptor("random", "random"),
        },
    )
    first = runner.execute(manifest)
    second = runner.execute(manifest)

    assert dumps_arena_result(first.result) == dumps_arena_result(second.result)
    assert tuple(game.game_id for game in first.result.games) == manifest.match_pairs[0].game_ids
    assert tuple(game.candidate_seat for game in first.result.games) == tuple(PlayerId)
    assert first.report.by_opponent[0].opponent_policy_id == "random"


def test_policy_routing_uses_each_planned_game_candidate_seat() -> None:
    pair = plan_match_pair("seat-routing", 44, "candidate", "opponent")

    assert ArenaRunner._policy_for(pair, pair.games[0], PlayerId.PLAYER_1) == "candidate"
    assert ArenaRunner._policy_for(pair, pair.games[0], PlayerId.PLAYER_2) == "opponent"
    assert ArenaRunner._policy_for(pair, pair.games[1], PlayerId.PLAYER_1) == "opponent"
    assert ArenaRunner._policy_for(pair, pair.games[1], PlayerId.PLAYER_2) == "candidate"


def _promotion_report(*, lower: float) -> tuple[PolicyDescriptor, ArenaManifest, ArenaReport]:
    candidate = PolicyDescriptor("candidate", "learned", "candidate-checkpoint")
    incumbent = PolicyDescriptor("incumbent", "learned", "incumbent-checkpoint")
    manifest = ArenaManifest(
        "promotion",
        candidate.policy_id,
        PolicyPool("incumbent-only", (PoolEntry(incumbent.policy_id, 1),)),
        tuple(
            plan_match_pair(f"seed-{seed}", seed, candidate.policy_id, incumbent.policy_id)
            for seed in range(PROMOTION_PAIR_COUNT)
        ),
        BootstrapConfig(seed=3, resamples=1),
    )
    games = tuple(
        ArenaGameResult(
            pair.pair_id,
            planned.game_id,
            planned.candidate_seat,
            TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,)),
            1,
        )
        for pair in manifest.match_pairs
        for planned in pair.games
    )
    report = build_arena_report(manifest, ArenaResult.for_manifest(manifest, games))
    return (
        candidate,
        manifest,
        replace(
            report,
            all_pairs=replace(
                report.all_pairs,
                confidence_interval=replace(report.all_pairs.confidence_interval, lower=lower),
            ),
        ),
    )


def test_promotion_bootstraps_then_requires_fixed_complete_arena_and_strict_lower_bound(
    tmp_path: Path,
) -> None:
    candidate, manifest, report = _promotion_report(lower=0.5)
    bootstrap = promotion_outcome(None, candidate, manifest, report)
    assert bootstrap.decision is PromotionDecision.BOOTSTRAPPED
    assert not bootstrap.champion.statistical_claim

    incumbent = ChampionManifest(
        "incumbent", "incumbent-checkpoint", None, PromotionDecision.BOOTSTRAPPED, False
    )
    retained = promotion_outcome(incumbent, candidate, manifest, report)
    assert retained.decision is PromotionDecision.RETAINED
    promoted = promotion_outcome(
        incumbent,
        candidate,
        manifest,
        replace(
            report,
            all_pairs=replace(
                report.all_pairs,
                confidence_interval=replace(report.all_pairs.confidence_interval, lower=0.500001),
            ),
        ),
    )
    assert promoted.decision is PromotionDecision.PROMOTED

    pointer = write_champion(tmp_path, bootstrap.champion)
    assert pointer.policy_id == "candidate"
    assert (
        loads_champion_manifest(dumps_champion_manifest(bootstrap.champion)) == bootstrap.champion
    )

    incomplete = replace(manifest, match_pairs=manifest.match_pairs[:-1])
    with pytest.raises(ArenaExecutionError, match="exactly"):
        promotion_outcome(incumbent, candidate, incomplete, report)
