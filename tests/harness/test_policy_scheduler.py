from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from innovation_ai.agents import RandomAgent
from innovation_ai.agents.descriptors import SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR
from innovation_ai.harness import (
    GameSpec,
    InnovationEngineAdapter,
    PendingGameDecision,
    PullGameRunner,
)
from innovation_ai.harness.policy import ValuePosition
from innovation_ai.harness.policy_scheduler import (
    LearnedPolicyAssignment,
    PolicyScheduler,
    SamplerFailure,
    SamplerFailureMode,
    SearchFailure,
    SearchPolicyAssignment,
)
from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.effects import load_effect_programs, start_dogma
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    build_explicit_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.search import (
    PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR,
    PRODUCTION_SEARCH_DESCRIPTOR,
    DeterministicSampledMinimax,
    DeterministicSampledRollout,
    MinimaxSelection,
    RolloutSearchDescriptor,
    RolloutSelection,
    SearchDescriptor,
    SearchRngFactory,
    seed_digest,
)
from innovation_ai.search.information_sets import (
    InformationSetSampler as SearchInformationSetSampler,
)
from innovation_ai.search.information_sets import InformationSetSpec as SearchInformationSetSpec
from innovation_ai.training.checkpoint import (
    DEFAULT_FALLBACK_AGENT_VERSION,
    DEFAULT_ROLLOUT_FALLBACK_AGENT,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    LEGACY_DEFAULT_FALLBACK_AGENT,
    LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    POLICY_DESCRIPTOR_SCHEMA_VERSION,
    PolicyDescriptor,
)
from innovation_ai.training.determinizations import InformationSetSampler, SamplingExhausted
from innovation_ai.training.selection import REPETITION_AWARE_SELECTOR_VERSION


@dataclass
class _ConstantEvaluator:
    calls: list[int]

    def evaluate(self, positions: Sequence[ValuePosition], /) -> tuple[float, ...]:
        self.calls.append(len(positions))
        return (0.5,) * len(positions)


class _FailingSampler:
    def sample_many(self, count: object, size: object) -> tuple[None, ...]:
        del count, size
        raise SamplingExhausted("fixture sampler failure")


@dataclass
class _StaticRunner:
    game_id: str
    authoritative_state: GameState
    decision: Decision

    def pending(self) -> tuple[PendingGameDecision, ...]:
        return (PendingGameDecision(self.game_id, self.decision),)

    def state(self, game_id: str) -> GameState:
        assert game_id == self.game_id
        return self.authoritative_state


class _FailingSearchSampler:
    def sample_many(self, spec: object, count: object) -> tuple[None, ...]:
        del spec, count
        raise RuntimeError("fixture search sampler failure")


class _ExplodingAgent:
    def choose_action(self, decision: Decision) -> SemanticAction:
        del decision
        raise AssertionError("strict search must not invoke an agent fallback")


def _legacy_policy() -> PolicyDescriptor:
    return PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        fallback_agent=LEGACY_DEFAULT_FALLBACK_AGENT,
        fallback_agent_version=DEFAULT_FALLBACK_AGENT_VERSION,
        search_descriptor_id=None,
        learned_turn_action_policy=None,
        search_continuation_policy=None,
        fallback_algorithm_kind=None,
        fallback_algorithm_descriptor_id=None,
        hypothetical_continuation_policy=None,
        schema_version=LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )


def _search_descriptor(*, determinizations: int = 1) -> SearchDescriptor:
    return SearchDescriptor(
        root_turn_horizon=1,
        opponent_turn_horizon=1,
        starting_meld_horizon=1,
        determinization_count=determinizations,
        route_transition_budget=1,
    )


def _rollout_descriptor(*, determinizations: int = 1) -> RolloutSearchDescriptor:
    return RolloutSearchDescriptor(determinization_count=determinizations)


def _policy(
    *,
    temperature: float = 0.0,
    selector_version: str = "temperature-softmax-v1",
    search_descriptor_id: str | None = None,
) -> PolicyDescriptor:
    return PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        temperature=temperature,
        selector_version=selector_version,
        search_descriptor_id=(
            PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
            if search_descriptor_id is None
            else search_descriptor_id
        ),
        search_continuation_policy=DEFAULT_SEARCH_CONTINUATION_POLICY,
        fallback_algorithm_kind=None,
        fallback_algorithm_descriptor_id=None,
        hypothetical_continuation_policy=None,
        schema_version=MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )


def _v3_policy(
    descriptor: SearchDescriptor | RolloutSearchDescriptor,
    *,
    hypothetical_continuation_policy: str | None = None,
) -> PolicyDescriptor:
    rollout = isinstance(descriptor, RolloutSearchDescriptor)
    hypothetical_policy = (
        descriptor.continuation_agent_descriptor_id
        if isinstance(descriptor, RolloutSearchDescriptor)
        else DEFAULT_SEARCH_CONTINUATION_POLICY
    )
    return PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        determinization_count=descriptor.determinization_count,
        fallback_agent=(DEFAULT_ROLLOUT_FALLBACK_AGENT if rollout else "sampled-minimax-heuristic"),
        fallback_algorithm_kind=descriptor.descriptor_kind,
        fallback_algorithm_descriptor_id=descriptor.descriptor_id,
        hypothetical_continuation_policy=(
            hypothetical_continuation_policy
            if hypothetical_continuation_policy is not None
            else hypothetical_policy
        ),
        schema_version=POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )


def _turn_runner(*game_ids: str) -> PullGameRunner[GameState]:
    runner = PullGameRunner(
        InnovationEngineAdapter(),
        tuple(GameSpec(game_id, 1100 + index) for index, game_id in enumerate(game_ids)),
    )
    bootstrap = PolicyScheduler({}, {}, run_seed=1, generation=0)
    setup = bootstrap.schedule(runner)
    assert all(audit.handling == "baseline" for audit in setup.audits)
    runner.submit(setup.submissions)
    return runner


def test_scheduler_flattens_turn_candidates_by_evaluator_key_and_returns_exact_actions() -> None:
    runner = _turn_runner("first", "second")
    low = _ConstantEvaluator([])
    high = _ConstantEvaluator([])
    descriptor = _policy()
    scheduler = PolicyScheduler(
        {
            "first": LearnedPolicyAssignment(descriptor, "low"),
            "second": LearnedPolicyAssignment(descriptor, "high"),
        },
        {"low": low, "high": high},
        run_seed=17,
        generation=2,
    )

    schedule = scheduler.schedule(runner)

    assert len(schedule.submissions) == len(runner.pending())
    assert all(
        audit.submission.action in request.decision.legal_actions
        for audit, request in zip(schedule.audits, runner.pending(), strict=True)
    )
    assert all(audit.handling == "learned" for audit in schedule.audits)
    assert len(schedule.selections) == 2
    # Different evaluator keys get separate calls; neither model receives the
    # other game's candidate positions.
    assert len(low.calls) == len(high.calls) == 1
    assert low.calls[0] > 0 and high.calls[0] > 0


def test_zero_temperature_and_stochastic_selection_are_rebatch_invariant() -> None:
    descriptor = _policy(temperature=0.25)

    def choose(game_ids: tuple[str, ...]) -> SemanticAction:
        runner = _turn_runner(*game_ids)
        evaluator = _ConstantEvaluator([])
        scheduler = PolicyScheduler(
            {game_id: LearnedPolicyAssignment(descriptor, "shared") for game_id in game_ids},
            {"shared": evaluator},
            run_seed=22,
            generation=4,
        )
        schedule = scheduler.schedule(runner)
        return next(item.action for item in schedule.submissions if item.game_id == "solo")

    assert choose(("solo",)) == choose(("solo", "other"))


def test_repetition_history_advances_only_after_commit_and_is_idempotent() -> None:
    runner = _turn_runner("history")
    descriptor = _policy(selector_version=REPETITION_AWARE_SELECTOR_VERSION)
    scheduler = PolicyScheduler(
        {"history": LearnedPolicyAssignment(descriptor, "value")},
        {"value": _ConstantEvaluator([])},
        run_seed=5,
        generation=0,
    )

    schedule = scheduler.schedule(runner)
    assert scheduler._recent_paid_actions == {}
    runner.submit(schedule.submissions)
    scheduler.record_committed(schedule)
    scheduler.record_committed(schedule)

    chooser = schedule.audits[0].chooser
    history = scheduler._recent_paid_actions[("history", chooser, descriptor.policy_id)]
    assert tuple(history) == (schedule.submissions[0].action,)


def test_sampler_failure_has_strict_and_heuristic_paths_without_true_state_fallback() -> None:
    runner = _turn_runner("failed")
    descriptor = _policy()

    def failing_factory(seed: bytes) -> InformationSetSampler:
        del seed
        return cast(InformationSetSampler, _FailingSampler())

    strict = PolicyScheduler(
        {"failed": LearnedPolicyAssignment(descriptor, "unused")},
        {},
        run_seed=3,
        generation=0,
        sampler_factory=failing_factory,
    )
    with pytest.raises(SamplerFailure, match="fixture sampler failure"):
        strict.schedule(runner)

    fallback = PolicyScheduler(
        {"failed": LearnedPolicyAssignment(descriptor, "unused")},
        {},
        run_seed=3,
        generation=0,
        sampler_failure_mode=SamplerFailureMode.HEURISTIC,
        sampler_factory=failing_factory,
    ).schedule(runner)
    assert fallback.audits[0].handling == "sampler-fallback"
    assert isinstance(fallback.audits[0].failure, SamplerFailure)
    assert fallback.submissions[0].action in runner.pending()[0].decision.legal_actions


def test_scheduler_supports_per_seat_learned_and_baseline_assignments() -> None:
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("mixed", 1200),))
    descriptor = _legacy_policy()
    scheduler = PolicyScheduler(
        {
            ("mixed", PlayerId.PLAYER_1): LearnedPolicyAssignment(descriptor, "learned"),
        },
        {"learned": _ConstantEvaluator([])},
        fallback_agents={("mixed", PlayerId.PLAYER_2): RandomAgent(44)},
        run_seed=9,
        generation=1,
    )

    setup = scheduler.schedule(runner)

    assert tuple(audit.handling for audit in setup.audits) == ("heuristic", "baseline")
    assert all(
        audit.submission.action in request.decision.legal_actions
        for audit, request in zip(setup.audits, runner.pending(), strict=True)
    )


def test_v2_learned_routes_starting_and_effect_choices_through_search() -> None:
    search_descriptor = _search_descriptor()
    learned = _policy(search_descriptor_id=search_descriptor.descriptor_id)
    setup_runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("setup-search", 1202),))
    scheduler = PolicyScheduler(
        {"setup-search": LearnedPolicyAssignment(learned, "unused")},
        {},
        run_seed=11,
        generation=2,
        search_descriptors={search_descriptor.descriptor_id: search_descriptor},
    )

    setup = scheduler.schedule(setup_runner)

    assert {audit.handling for audit in setup.audits} == {"learned-search-fallback"}
    assert len(setup.search_selections) == 2
    assert setup.selections == ()
    assert all(
        selection.telemetry.selector_seed_digest is not None
        for selection in setup.search_selections
    )

    effect_state = build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("writing"),),
                    board=((Color.YELLOW, (CardId("agriculture"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.RED, (CardId("archery"),)),)),
            ),
        )
    )
    paused = start_dogma(
        effect_state,
        CardId("agriculture"),
        PlayerId.PLAYER_1,
        load_effect_programs(),
    )
    assert paused.decision is not None
    effect_runner = _StaticRunner("effect-search", paused.state, paused.decision)
    effect = PolicyScheduler(
        {"effect-search": LearnedPolicyAssignment(learned, "unused")},
        {},
        run_seed=11,
        generation=2,
        search_descriptors={search_descriptor.descriptor_id: search_descriptor},
    ).schedule(cast(PullGameRunner[GameState], effect_runner))

    assert effect.audits[0].handling == "learned-search-fallback"
    assert effect.audits[0].search_selection is not None
    assert effect.submissions[0].action in paused.decision.legal_actions


def test_schema_v2_learned_policy_cannot_reinterpret_rollout_descriptor() -> None:
    rollout = _rollout_descriptor()
    learned = _policy(search_descriptor_id=rollout.descriptor_id)
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("learned-rollout", 1203),))
    scheduler = PolicyScheduler(
        {"learned-rollout": LearnedPolicyAssignment(learned, "unused")},
        {},
        run_seed=11,
        generation=2,
        search_descriptors={rollout.descriptor_id: rollout},
    )

    with pytest.raises(SearchFailure, match="schema-v2 learned policy requires a minimax"):
        scheduler.schedule(runner)


def test_schema_v3_learned_fallback_dispatches_registered_minimax_and_rollout() -> None:
    minimax = _search_descriptor()
    rollout = _rollout_descriptor()
    for game_id, descriptor, expected_type in (
        ("learned-v3-minimax", minimax, MinimaxSelection),
        ("learned-v3-rollout", rollout, RolloutSelection),
    ):
        learned = _v3_policy(descriptor)
        runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec(game_id, 1204),))
        scheduler = PolicyScheduler(
            {game_id: LearnedPolicyAssignment(learned, "unused")},
            {},
            run_seed=13,
            generation=2,
            search_descriptors={descriptor.descriptor_id: descriptor},
            heuristic=_ExplodingAgent(),
        )

        schedule = scheduler.schedule(runner)

        assert all(audit.handling == "learned-search-fallback" for audit in schedule.audits)
        assert all(isinstance(selection, expected_type) for selection in schedule.search_selections)


def test_schema_v3_learned_fallback_rejects_kind_and_hypothetical_policy_mismatches() -> None:
    minimax = _search_descriptor()
    rollout = _rollout_descriptor()
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("v3-mismatch", 1205),))
    wrong_kind = PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
        fallback_algorithm_kind=rollout.descriptor_kind,
        fallback_algorithm_descriptor_id=minimax.descriptor_id,
        hypothetical_continuation_policy=rollout.continuation_agent_descriptor_id,
    )
    scheduler = PolicyScheduler(
        {"v3-mismatch": LearnedPolicyAssignment(wrong_kind, "unused")},
        {},
        run_seed=13,
        generation=2,
        search_descriptors={minimax.descriptor_id: minimax},
    )
    with pytest.raises(SearchFailure, match="algorithm kind differs"):
        scheduler.schedule(runner)

    wrong_policy = _v3_policy(rollout, hypothetical_continuation_policy="wrong-policy-v1")
    scheduler = PolicyScheduler(
        {"v3-mismatch": LearnedPolicyAssignment(wrong_policy, "unused")},
        {},
        run_seed=13,
        generation=2,
        search_descriptors={rollout.descriptor_id: rollout},
    )
    with pytest.raises(SearchFailure, match="hypothetical continuation policy differs"):
        scheduler.schedule(runner)


def test_explicit_search_assignment_controls_every_decision_kind_and_is_rebatch_invariant() -> None:
    descriptor = _search_descriptor()
    assignment = SearchPolicyAssignment("search-policy", descriptor)

    def setup_choice(game_ids: tuple[str, ...]) -> tuple[SemanticAction, str | None]:
        runner = PullGameRunner(
            InnovationEngineAdapter(),
            tuple(GameSpec(game_id, 1400 + index) for index, game_id in enumerate(game_ids)),
        )
        schedule = PolicyScheduler(
            {},
            {},
            run_seed=19,
            generation=3,
            search_assignments={game_id: assignment for game_id in game_ids},
            heuristic=_ExplodingAgent(),
        ).schedule(runner)
        assert all(audit.handling == "search" for audit in schedule.audits)
        submission = next(item for item in schedule.submissions if item.game_id == "solo")
        audit = next(item for item in schedule.audits if item.submission == submission)
        assert audit.search_selection is not None
        return submission.action, audit.search_selection.telemetry.selector_seed_digest

    assert setup_choice(("solo",)) == setup_choice(("solo", "other"))

    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("all-search", 1410),))
    scheduler = PolicyScheduler(
        {},
        {},
        run_seed=19,
        generation=3,
        search_assignments={"all-search": assignment},
        heuristic=_ExplodingAgent(),
    )
    setup = scheduler.schedule(runner)
    runner.submit(setup.submissions)
    turn = scheduler.schedule(runner)
    assert all(audit.handling == "search" for audit in turn.audits)
    assert all(audit.search_selection is not None for audit in turn.audits)


def test_explicit_rollout_assignment_uses_rollout_result_and_sampler_seed_telemetry() -> None:
    descriptor = PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR
    assignment = SearchPolicyAssignment(
        SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR.descriptor_id,
        descriptor,
    )
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("rollout", 1409),))
    scheduler = PolicyScheduler(
        {},
        {},
        run_seed=19,
        generation=3,
        search_assignments={"rollout": assignment},
        heuristic=_ExplodingAgent(),
    )

    pending = runner.pending()
    schedule = scheduler.schedule(runner)

    assert all(audit.handling == "search" for audit in schedule.audits)
    assert all(isinstance(selection, RolloutSelection) for selection in schedule.search_selections)
    for request, selection in zip(pending, schedule.search_selections, strict=True):
        assert isinstance(selection, RolloutSelection)
        sampler_seed = SearchRngFactory(19, 3).seed_for_decision(
            game_id=request.game_id,
            chooser=request.decision.chooser,
            decision_id=request.decision.decision_id,
            policy_id=assignment.policy_id,
            search_descriptor_id=descriptor.descriptor_id,
        )
        assert selection.telemetry.sampler_seed_digest == seed_digest(sampler_seed)
        assert not hasattr(selection.telemetry, "selector_seed_digest")
    assert isinstance(scheduler._searchers[descriptor.descriptor_id], DeterministicSampledRollout)


def test_search_dispatch_is_strict_and_keeps_minimax_legacy_seed_telemetry() -> None:
    minimax = _search_descriptor()
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("minimax", 1411),))
    scheduler = PolicyScheduler(
        {},
        {},
        run_seed=23,
        generation=4,
        search_assignments={"minimax": SearchPolicyAssignment("minimax-policy", minimax)},
    )

    pending = runner.pending()
    schedule = scheduler.schedule(runner)

    for request, selection in zip(pending, schedule.search_selections, strict=True):
        assert not isinstance(selection, RolloutSelection)
        sampler_seed = SearchRngFactory(23, 4).seed_for_decision(
            game_id=request.game_id,
            chooser=request.decision.chooser,
            decision_id=request.decision.decision_id,
            policy_id="minimax-policy",
            search_descriptor_id=minimax.descriptor_id,
        )
        assert selection.telemetry.selector_seed_digest == seed_digest(sampler_seed)
    assert isinstance(scheduler._searchers[minimax.descriptor_id], DeterministicSampledMinimax)
    with pytest.raises(ValueError, match="unsupported minimax search algorithm"):
        SearchPolicyAssignment(
            "wrong-minimax",
            SearchDescriptor(search_version="root-sampled-deterministic-turn-rollout-v1"),
        )
    with pytest.raises(ValueError, match="unsupported rollout search algorithm"):
        SearchPolicyAssignment(
            "wrong-rollout",
            RolloutSearchDescriptor(algorithm="root-sampled-minimax-one-completed-turn-v1"),
        )


def test_search_builder_receives_only_live_state_and_exact_decision_boundary() -> None:
    descriptor = _search_descriptor()
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("safe-builder", 1500),))
    scheduler = PolicyScheduler(
        {},
        {},
        run_seed=2,
        generation=0,
        search_assignments={"safe-builder": SearchPolicyAssignment("search-policy", descriptor)},
    )
    actual_builder = scheduler._search_spec_builder
    calls: list[tuple[GameState, Decision]] = []

    class _RecordingBuilder:
        def build(self, state: GameState, decision: Decision) -> SearchInformationSetSpec:
            calls.append((state, decision))
            return actual_builder.build(state, decision)

    scheduler._search_spec_builder = _RecordingBuilder()  # type: ignore[assignment]
    pending = runner.pending()
    scheduler.schedule(runner)

    assert calls == [(runner.state(request.game_id), request.decision) for request in pending]


def test_search_failure_is_always_strict_and_never_invokes_simple_fallback() -> None:
    descriptor = _search_descriptor()
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("strict-search", 1600),))

    def failing_factory(seed: bytes) -> SearchInformationSetSampler:
        del seed
        return cast(SearchInformationSetSampler, _FailingSearchSampler())

    scheduler = PolicyScheduler(
        {},
        {},
        run_seed=3,
        generation=0,
        search_assignments={"strict-search": SearchPolicyAssignment("search-policy", descriptor)},
        sampler_failure_mode=SamplerFailureMode.HEURISTIC,
        heuristic=_ExplodingAgent(),
        search_sampler_factory=failing_factory,
    )

    with pytest.raises(SearchFailure, match="fixture search sampler failure"):
        scheduler.schedule(runner)


def test_schema_v1_learned_setup_keeps_original_simple_heuristic_fallback() -> None:
    runner = PullGameRunner(InnovationEngineAdapter(), (GameSpec("legacy", 1700),))
    descriptor = _legacy_policy()
    scheduler = PolicyScheduler(
        {"legacy": LearnedPolicyAssignment(descriptor, "learned")},
        {"learned": _ConstantEvaluator([])},
        fallback_agents={
            ("legacy", PlayerId.PLAYER_1): _ExplodingAgent(),
            ("legacy", PlayerId.PLAYER_2): _ExplodingAgent(),
        },
        run_seed=8,
        generation=0,
    )

    schedule = scheduler.schedule(runner)

    assert all(audit.handling == "heuristic" for audit in schedule.audits)
    assert schedule.search_selections == ()
    runner.submit(schedule.submissions)
    turn = scheduler.schedule(runner)
    assert all(audit.handling == "learned" for audit in turn.audits)
    assert turn.search_selections == ()
    assert len(turn.selections) == 1
