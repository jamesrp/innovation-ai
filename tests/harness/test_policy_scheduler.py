from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from innovation_ai.agents import RandomAgent
from innovation_ai.harness import GameSpec, InnovationEngineAdapter, PullGameRunner
from innovation_ai.harness.policy import ValuePosition
from innovation_ai.harness.policy_scheduler import (
    LearnedPolicyAssignment,
    PolicyScheduler,
    SamplerFailure,
    SamplerFailureMode,
)
from innovation_ai.innovation.actions import SemanticAction
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.checkpoint import PolicyDescriptor
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


def _policy(
    *, temperature: float = 0.0, selector_version: str = "temperature-softmax-v1"
) -> PolicyDescriptor:
    return PolicyDescriptor(
        checkpoint_id="fixture-checkpoint",
        encoder_layout_fingerprint="fixture-encoder",
        card_data_fingerprint="fixture-cards",
        effects_fingerprint="fixture-effects",
        temperature=temperature,
        selector_version=selector_version,
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
    descriptor = _policy()
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
