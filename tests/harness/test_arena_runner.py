from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from innovation_ai.agents.descriptors import (
    SAMPLED_MINIMAX_AGENT_DESCRIPTOR,
    SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
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
    ArenaActionLimitError,
    ArenaExecutionError,
    ArenaRunner,
    ArenaTraceConfiguration,
    ChampionManifest,
    PromotionDecision,
    dumps_champion_manifest,
    loads_champion_manifest,
    promotion_outcome,
    write_champion,
)
from innovation_ai.harness.diagnostics import read_diagnostic_trace
from innovation_ai.harness.engine import InnovationEngineAdapter
from innovation_ai.innovation.state import TerminalReason, TerminalResult
from innovation_ai.innovation.types import PlayerId
from innovation_ai.search import (
    MINIMAX_SEARCH_DESCRIPTOR_KIND,
    ROLLOUT_SEARCH_DESCRIPTOR_KIND,
    RolloutSearchDescriptor,
    SearchDescriptor,
)
from innovation_ai.training.checkpoint import (
    DEFAULT_ROLLOUT_FALLBACK_AGENT,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
)
from innovation_ai.training.checkpoint import (
    PolicyDescriptor as TrainingPolicyDescriptor,
)


def _canonical_search_policy_id(descriptor: SearchDescriptor | RolloutSearchDescriptor) -> str:
    family = (
        SAMPLED_MINIMAX_AGENT_DESCRIPTOR
        if isinstance(descriptor, SearchDescriptor)
        else SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR
    )
    return AgentDescriptor(
        family.name,
        family.version,
        (("search_descriptor_id", descriptor.descriptor_id),),
    ).descriptor_id


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
    assert first.search_telemetry.decisions == 0
    assert first.search_telemetry.routes == 0


def test_executor_writes_deterministic_exact_private_trace_chain_and_redacted_summary(
    tmp_path: Path,
) -> None:
    manifest = _baseline_manifest()
    policies = {
        "heuristic": PolicyDescriptor("heuristic", "heuristic"),
        "random": PolicyDescriptor("random", "random"),
    }
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = ArenaRunner(
        InnovationEngineAdapter(),
        policies,
        trace_configuration=ArenaTraceConfiguration(first_root, "revision-1"),
    ).execute(manifest)
    second = ArenaRunner(
        InnovationEngineAdapter(),
        policies,
        trace_configuration=ArenaTraceConfiguration(second_root, "revision-1"),
    ).execute(manifest)

    assert set(first.trace_artifacts) == set(manifest.match_pairs[0].game_ids)
    for game in first.result.games:
        first_artifact = first.trace_artifacts[game.game_id]
        second_artifact = second.trace_artifacts[game.game_id]
        assert first_artifact.private_path.read_bytes() == second_artifact.private_path.read_bytes()
        assert (
            first_artifact.redacted_path.read_bytes() == second_artifact.redacted_path.read_bytes()
        )
        assert first_artifact.private_sha256 == second_artifact.private_sha256
        assert first_artifact.redacted_sha256 == second_artifact.redacted_sha256

        trace = read_diagnostic_trace(first_artifact.private_path)
        assert trace.footer.step_count == game.game_length
        assert len(trace.steps) == game.game_length
        assert trace.footer.failure is None
        redacted = first_artifact.redacted_path.read_text("ascii")
        assert '"legal_actions"' not in redacted
        assert '"selected_action"' not in redacted
        assert '"actions"' not in redacted


def test_action_ceiling_writes_failure_trace_before_raising(tmp_path: Path) -> None:
    manifest = _baseline_manifest()
    failing_game_id = manifest.match_pairs[0].games[0].game_id
    runner = ArenaRunner(
        InnovationEngineAdapter(),
        {
            "heuristic": PolicyDescriptor("heuristic", "heuristic"),
            "random": PolicyDescriptor("random", "random"),
        },
        max_actions=2,
        trace_configuration=ArenaTraceConfiguration(tmp_path, "revision-1"),
    )

    with pytest.raises(ArenaActionLimitError):
        runner.execute(manifest)

    trace = read_diagnostic_trace(tmp_path / f"{failing_game_id}.private.jsonl.gz")
    assert trace.footer.outcome == "action-ceiling"
    assert trace.footer.failure is not None
    assert trace.footer.step_count == 2
    assert (tmp_path / f"{failing_game_id}.redacted.json").is_file()


def test_arena_trace_snapshot_markers_are_explicit_and_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private-debug"):
        ArenaTraceConfiguration(
            tmp_path,
            "revision-1",
            authoritative_snapshots=True,
        )

    manifest = _baseline_manifest()
    execution = ArenaRunner(
        InnovationEngineAdapter(),
        {
            "heuristic": PolicyDescriptor("heuristic", "heuristic"),
            "random": PolicyDescriptor("random", "random"),
        },
        trace_configuration=ArenaTraceConfiguration(
            tmp_path,
            "revision-1",
            authoritative_snapshots=True,
            private_debug=True,
        ),
    ).execute(manifest)
    trace = read_diagnostic_trace(next(iter(execution.trace_artifacts.values())).private_path)
    assert trace.header.authoritative_snapshots
    assert trace.header.private_debug
    assert all(step.before_snapshot is not None for step in trace.steps)
    assert all(step.after_snapshot is not None for step in trace.steps)


def _tiny_search_descriptor() -> SearchDescriptor:
    return SearchDescriptor(
        root_turn_horizon=1,
        opponent_turn_horizon=1,
        starting_meld_horizon=1,
        determinization_count=1,
        route_transition_budget=1,
    )


def test_search_heuristic_runs_every_decision_through_scheduler_and_aggregates_telemetry() -> None:
    search = _tiny_search_descriptor()
    search_policy_id = _canonical_search_policy_id(search)
    manifest = ArenaManifest(
        "search-execution",
        search_policy_id,
        PolicyPool("simple-only", (PoolEntry("simple", 1),)),
        (plan_match_pair("search-seed-808", 808, search_policy_id, "simple"),),
        BootstrapConfig(seed=7, resamples=3),
    )
    execution = ArenaRunner(
        InnovationEngineAdapter(),
        {
            search_policy_id: PolicyDescriptor(search_policy_id, "search-heuristic"),
            "simple": PolicyDescriptor("simple", "simple-heuristic"),
        },
        search_policy_descriptors={search_policy_id: search},
        search_descriptors={search.descriptor_id: search},
    ).execute(manifest)

    assert len(execution.result.games) == 2
    assert execution.search_telemetry.decisions > 2
    assert execution.search_telemetry.routes >= execution.search_telemetry.decisions
    assert execution.search_telemetry.root_transitions == execution.search_telemetry.routes
    assert execution.search_telemetry.recursive_engine_transitions >= 0
    assert execution.search_telemetry.mandatory_setup_transitions >= 0
    assert execution.search_telemetry.transposition_hits >= 0
    assert execution.search_telemetry.repeated_position_cutoffs >= 0
    assert execution.search_telemetry.budget_cutoff_routes >= 0
    assert execution.search_telemetry.immediate_leaf_fallback_routes >= 0
    with pytest.raises(FrozenInstanceError):
        execution.search_telemetry.routes = 0  # type: ignore[misc]


def test_search_policy_requires_registered_content_derived_descriptor_identity() -> None:
    policy = {"search": PolicyDescriptor("search", "search-heuristic")}
    search = _tiny_search_descriptor()

    with pytest.raises(ArenaExecutionError, match="no search descriptor"):
        ArenaRunner(InnovationEngineAdapter(), policy)
    with pytest.raises(ArenaExecutionError, match="identity is unavailable"):
        ArenaRunner(
            InnovationEngineAdapter(),
            policy,
            search_policy_descriptors={"search": search},
        )
    with pytest.raises(ArenaExecutionError, match="not the canonical"):
        ArenaRunner(
            InnovationEngineAdapter(),
            policy,
            search_policy_descriptors={"search": search},
            search_descriptors={search.descriptor_id: search},
        )

    canonical_id = _canonical_search_policy_id(search)
    with pytest.raises(ArenaExecutionError, match="cannot reference a learned checkpoint"):
        ArenaRunner(
            InnovationEngineAdapter(),
            {
                canonical_id: PolicyDescriptor(
                    canonical_id,
                    "search-heuristic",
                    "not-a-baseline-checkpoint",
                )
            },
            search_policy_descriptors={canonical_id: search},
            search_descriptors={search.descriptor_id: search},
        )


def test_arena_accepts_rollout_descriptors_and_rejects_algorithm_family_mismatches(
    tmp_path: Path,
) -> None:
    rollout = RolloutSearchDescriptor(determinization_count=1)
    rollout_policy_id = _canonical_search_policy_id(rollout)
    policy = {rollout_policy_id: PolicyDescriptor(rollout_policy_id, "search-heuristic")}

    runner = ArenaRunner(
        InnovationEngineAdapter(),
        policy,
        search_policy_descriptors={rollout_policy_id: rollout},
        search_descriptors={rollout.descriptor_id: rollout},
    )
    assert runner._search_policy_descriptors[rollout_policy_id] == rollout

    manifest = ArenaManifest(
        "rollout-routing",
        rollout_policy_id,
        PolicyPool("simple-only", (PoolEntry("simple", 1),)),
        (plan_match_pair("rollout-routing-pair", 812, rollout_policy_id, "simple"),),
        BootstrapConfig(seed=7, resamples=1),
    )
    traced_runner = ArenaRunner(
        InnovationEngineAdapter(),
        {
            rollout_policy_id: PolicyDescriptor(rollout_policy_id, "search-heuristic"),
            "simple": PolicyDescriptor("simple", "simple-heuristic"),
        },
        search_policy_descriptors={rollout_policy_id: rollout},
        search_descriptors={rollout.descriptor_id: rollout},
        max_actions=1,
        trace_configuration=ArenaTraceConfiguration(tmp_path, "revision-rollout"),
    )
    with pytest.raises(ArenaActionLimitError):
        traced_runner.execute(manifest)
    trace = read_diagnostic_trace(
        tmp_path / f"{manifest.match_pairs[0].games[0].game_id}.private.jsonl.gz"
    )
    rollout_steps = [step for step in trace.steps if step.search is not None]
    assert rollout_steps
    search_payload = rollout_steps[0].search
    assert search_payload is not None
    telemetry = search_payload["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["rollout_descriptor_id"] == rollout.descriptor_id
    assert "search_descriptor_id" not in telemetry

    with pytest.raises(ArenaExecutionError, match="algorithm is unsupported"):
        invalid = RolloutSearchDescriptor(algorithm="root-sampled-minimax-one-completed-turn-v1")
        ArenaRunner(
            InnovationEngineAdapter(),
            policy,
            search_policy_descriptors={rollout_policy_id: invalid},
            search_descriptors={invalid.descriptor_id: invalid},
        )

    learned = TrainingPolicyDescriptor(
        checkpoint_id="rollout-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        determinization_count=rollout.determinization_count,
        fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
        fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
        fallback_algorithm_descriptor_id=rollout.descriptor_id,
        hypothetical_continuation_policy=rollout.continuation_agent_descriptor_id,
    )

    class _Cache:
        def evaluator_for(self, descriptor: TrainingPolicyDescriptor) -> object:
            del descriptor
            return object()

    learned_policy = {"learned": PolicyDescriptor("learned", "learned", "rollout-checkpoint")}
    ArenaRunner(
        InnovationEngineAdapter(),
        learned_policy,
        learned_policies={"learned": learned},
        evaluator_cache=_Cache(),  # type: ignore[arg-type]
        search_descriptors={rollout.descriptor_id: rollout},
    )
    confused = replace(
        learned,
        fallback_algorithm_kind=MINIMAX_SEARCH_DESCRIPTOR_KIND,
        fallback_agent="sampled-minimax-heuristic",
    )
    with pytest.raises(ArenaExecutionError, match="fallback algorithm kind is incompatible"):
        ArenaRunner(
            InnovationEngineAdapter(),
            learned_policy,
            learned_policies={"learned": confused},
            evaluator_cache=_Cache(),  # type: ignore[arg-type]
            search_descriptors={rollout.descriptor_id: rollout},
        )


def test_learned_v2_setup_fallback_receives_arena_search_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from innovation_ai.harness.policy_scheduler import PolicyScheduler

    search = _tiny_search_descriptor()
    learned = TrainingPolicyDescriptor(
        checkpoint_id="learned-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        search_descriptor_id=search.descriptor_id,
        search_continuation_policy=DEFAULT_SEARCH_CONTINUATION_POLICY,
        fallback_algorithm_kind=None,
        fallback_algorithm_descriptor_id=None,
        hypothetical_continuation_policy=None,
        schema_version=MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )

    class _UnusedEvaluator:
        def evaluate(self, positions: object) -> tuple[float, ...]:
            del positions
            raise AssertionError("setup fallback must not request learned values")

    class _UnusedCache:
        def evaluator_for(self, descriptor: TrainingPolicyDescriptor) -> _UnusedEvaluator:
            del descriptor
            return _UnusedEvaluator()

    handlings: list[str] = []
    original_schedule = PolicyScheduler.schedule

    def recording_schedule(self: PolicyScheduler, runner: object) -> Any:
        schedule = original_schedule(self, runner)  # type: ignore[arg-type]
        handlings.extend(audit.handling for audit in schedule.audits)
        return schedule

    monkeypatch.setattr(PolicyScheduler, "schedule", recording_schedule)
    manifest = ArenaManifest(
        "learned-search-fallback",
        "learned",
        PolicyPool("simple-only", (PoolEntry("simple", 1),)),
        (plan_match_pair("learned-search-seed", 809, "learned", "simple"),),
        BootstrapConfig(seed=7, resamples=3),
    )
    runner = ArenaRunner(
        InnovationEngineAdapter(),
        {
            "learned": PolicyDescriptor("learned", "learned", "learned-checkpoint"),
            "simple": PolicyDescriptor("simple", "simple-heuristic"),
        },
        learned_policies={"learned": learned},
        evaluator_cache=_UnusedCache(),  # type: ignore[arg-type]
        search_descriptors={search.descriptor_id: search},
        max_actions=1,
    )

    with pytest.raises(ArenaActionLimitError):
        runner.execute(manifest)
    assert "learned-search-fallback" in handlings


def test_action_ceiling_retains_reproducible_diagnostic() -> None:
    manifest = _baseline_manifest()
    runner = ArenaRunner(
        InnovationEngineAdapter(),
        {
            "heuristic": PolicyDescriptor("heuristic", "heuristic"),
            "random": PolicyDescriptor("random", "random"),
        },
        max_actions=2,
    )

    with pytest.raises(ArenaActionLimitError) as caught:
        runner.execute(manifest)

    diagnostic = caught.value.diagnostic
    assert diagnostic["format"] == "innovation-ai-arena-action-ceiling-failure"
    assert diagnostic["game_id"] == manifest.match_pairs[0].games[0].game_id
    assert diagnostic["setup_seed"] == manifest.match_pairs[0].setup_seed
    assert diagnostic["action_count"] == 2
    assert diagnostic["action_ceiling"] == 2
    assert str(diagnostic["current_state_hash"]).startswith("sha256:")
    action_tail = diagnostic["action_tail"]
    assert isinstance(action_tail, list)
    assert len(action_tail) == 2


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
