from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO

import pytest

from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
from innovation_ai.harness.actor_pool import BoundedActorPool
from innovation_ai.harness.benchmark import (
    BaselineBenchmarkConfig,
    BaselineScenario,
    run_baseline_scenario,
)
from innovation_ai.harness.metrics import JsonlMetricSink, NoOpMetricSink, NoOpTimerSink
from innovation_ai.harness.records import (
    GameResult,
    RunnerRecording,
    SemanticActionEvent,
    SemanticActionSink,
)
from innovation_ai.harness.runner import GameSpec, PullGameRunner, Submission
from innovation_ai.harness.seeds import agent_seed, derive_seed, setup_seed
from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import (
    GameState,
    TerminalReason,
    TerminalResult,
    build_setup_state,
    state_hash,
)
from innovation_ai.innovation.types import PlayerId
from innovation_ai.innovation.zones import ValidationLevel


@dataclass(frozen=True, slots=True)
class _State:
    game: GameState
    action_count: int
    terminal: TerminalResult | None = None


class _ThreeActionEngine:
    """Exercise pull boundaries without retaining a full production game to termination."""

    def initial_state(self, seed: int, /) -> _State:
        return _State(build_setup_state(seed), 0)

    def pending_decisions(self, state: _State, /) -> tuple[Decision, ...]:
        return () if state.terminal is not None else current_decisions(state.game)

    def apply(self, state: _State, action: SemanticAction, /) -> _State:
        game = apply_action(state.game, action).state
        count = state.action_count + 1
        terminal = (
            TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,)) if count == 3 else None
        )
        return _State(game, count, terminal)

    def terminal_result(self, state: _State, /) -> TerminalResult | None:
        return state.terminal

    def fingerprint(self, state: _State, /) -> str:
        suffix = state.terminal.reason.value if state.terminal is not None else "active"
        return f"{state_hash(state.game)}:{state.action_count}:{suffix}"


class _ActionCollector(SemanticActionSink):
    def __init__(self) -> None:
        self.events: list[SemanticActionEvent] = []

    def record_action(self, event: SemanticActionEvent, /) -> None:
        self.events.append(event)


def _drive(runner: PullGameRunner[_State]) -> GameResult[_State]:
    while True:
        result = runner.result("game")
        if result is not None:
            return result
        submissions = tuple(
            Submission(item.game_id, item.decision.legal_actions[0]) for item in runner.pending()
        )
        runner.submit(submissions)


def test_agent_descriptors_and_domain_separated_seeds_are_stable() -> None:
    descriptor = AgentDescriptor("test", "v1", (("a", 1), ("enabled", True)))
    assert descriptor.canonical_json() == descriptor.canonical_json()
    assert descriptor.descriptor_id == descriptor.descriptor_id
    with pytest.raises(ValueError, match="sorted"):
        AgentDescriptor("test", "v1", (("z", 1), ("a", 2)))

    first = derive_seed(77, "test/domain", "game", 3)
    assert first == derive_seed(77, "test/domain", "game", 3)
    assert first != derive_seed(77, "other/domain", "game", 3)
    assert setup_seed(77, "game") != agent_seed(77, "game", "player-1", "random-v1")


def test_metric_sinks_are_noop_or_canonical_jsonl() -> None:
    NoOpMetricSink().record("actors.completed", 2, scenario="fixture")
    with NoOpTimerSink().timer("actors.wait", scenario="fixture"):
        pass

    stream = StringIO()
    sink = JsonlMetricSink(stream)
    sink.record("actors.completed", 2, scenario="fixture")
    with sink.timer("actors.wait", scenario="fixture"):
        pass
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0] == {
        "fields": {"scenario": "fixture"},
        "kind": "metric",
        "name": "actors.completed",
        "value": 2,
    }
    assert rows[1]["fields"] == {"scenario": "fixture", "unit": "seconds"}
    assert rows[1]["name"] == "actors.wait"


def test_compact_action_events_match_default_audit_actions_and_final_result() -> None:
    default = _drive(PullGameRunner(_ThreeActionEngine(), (GameSpec("game", 100),)))
    collector = _ActionCollector()
    compact = _drive(
        PullGameRunner(
            _ThreeActionEngine(),
            (GameSpec("game", 100),),
            recording=RunnerRecording(
                retain_actions=False,
                transition_fingerprints=False,
                action_sink=collector,
            ),
        )
    )

    assert tuple(event.action for event in collector.events) == tuple(
        action.action for action in default.record.actions
    )
    assert tuple(event.decision_id for event in collector.events) == tuple(
        action.decision_id for action in default.record.actions
    )
    assert all(event.resulting_state is None for event in collector.events)
    assert compact.record.actions == ()
    assert compact.record.initial_state == default.record.initial_state
    assert compact.record.final_state == default.record.final_state
    assert compact.record.terminal == default.record.terminal


def test_bounded_actor_pool_retires_records_before_refilling_slots() -> None:
    completed: list[str] = []
    pool = BoundedActorPool(
        _ThreeActionEngine(),
        tuple(GameSpec(f"game-{index}", 200 + index) for index in range(5)),
        max_games_in_flight=2,
        on_complete=lambda result: completed.append(result.record.game_id),
        recording=RunnerRecording(retain_actions=False, transition_fingerprints=False),
    )
    assert pool.in_flight_count == 2
    while not pool.is_finished:
        pending = pool.pending()
        assert pending
        pool.submit(
            tuple(Submission(item.game_id, item.decision.legal_actions[0]) for item in pending)
        )
        assert pool.in_flight_count <= pool.max_games_in_flight

    assert completed == [f"game-{index}" for index in range(5)]
    assert pool.completed_count == 5
    assert pool.game_ids == ()
    assert not hasattr(pool, "results")


def test_baseline_scenario_is_batching_independent_and_reports_environment_metrics() -> None:
    scenario = BaselineScenario(
        "fixture",
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
        RANDOM_AGENT_DESCRIPTOR,
    )
    sequential = run_baseline_scenario(
        _ThreeActionEngine(),
        scenario,
        BaselineBenchmarkConfig(
            run_seed=300,
            games_per_scenario=4,
            games_in_flight=1,
            validation_levels=(ValidationLevel.CHEAP,),
        ),
        ValidationLevel.CHEAP,
    )
    batched = run_baseline_scenario(
        _ThreeActionEngine(),
        scenario,
        BaselineBenchmarkConfig(
            run_seed=300,
            games_per_scenario=4,
            games_in_flight=2,
            validation_levels=(ValidationLevel.CHEAP,),
        ),
        ValidationLevel.CHEAP,
    )

    assert sequential.semantic_digest == batched.semantic_digest
    assert sequential.actions == batched.actions == 12
    assert sequential.terminal_reasons == ((TerminalReason.CARD_EFFECT.value, 4),)
    assert sequential.games_per_second >= 0
    assert sequential.peak_rss_bytes >= 0
