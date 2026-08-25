from __future__ import annotations

from dataclasses import dataclass

import pytest

from innovation_ai.agents import Agent, RandomAgent, ScriptedAgent
from innovation_ai.harness import (
    DuplicateGameError,
    GameSpec,
    InnovationEngineAdapter,
    PullGameRunner,
    RunnerError,
    SingleGameRunner,
    StepLimitError,
    Submission,
    SubmissionError,
    UnknownGameError,
)
from innovation_ai.harness.engine import RunnerEngine
from innovation_ai.innovation.actions import Decision, DogmaAction, DrawAction, SemanticAction
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import (
    GameState,
    TerminalReason,
    TerminalResult,
    build_setup_state,
    state_hash,
)
from innovation_ai.innovation.types import PlayerId


@dataclass(frozen=True, slots=True)
class SyntheticState:
    game: GameState
    action_count: int
    terminal: TerminalResult | None = None


class SyntheticTerminalEngine:
    """Freeze A setup/turn protocol with a synthetic three-action terminal boundary."""

    def initial_state(self, seed: int, /) -> SyntheticState:
        return SyntheticState(build_setup_state(seed), 0)

    def pending_decisions(self, state: SyntheticState, /) -> tuple[Decision, ...]:
        return () if state.terminal is not None else current_decisions(state.game)

    def apply(self, state: SyntheticState, action: SemanticAction, /) -> SyntheticState:
        transitioned = apply_action(state.game, action).state
        count = state.action_count + 1
        terminal = (
            TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,)) if count == 3 else None
        )
        return SyntheticState(transitioned, count, terminal)

    def terminal_result(self, state: SyntheticState, /) -> TerminalResult | None:
        return state.terminal

    def fingerprint(self, state: SyntheticState, /) -> str:
        suffix = state.terminal.reason.value if state.terminal is not None else "active"
        return f"{state_hash(state.game)}:{state.action_count}:{suffix}"


class InitiallyTerminalEngine(SyntheticTerminalEngine):
    def initial_state(self, seed: int, /) -> SyntheticState:
        return SyntheticState(
            build_setup_state(seed),
            0,
            TerminalResult(TerminalReason.CARD_EFFECT),
        )


class DuplicateDecisionEngine(SyntheticTerminalEngine):
    def pending_decisions(self, state: SyntheticState, /) -> tuple[Decision, ...]:
        decisions = super().pending_decisions(state)
        return (decisions[0], decisions[0]) if decisions else ()


class TerminalAfterOneEngine(SyntheticTerminalEngine):
    def apply(self, state: SyntheticState, action: SemanticAction, /) -> SyntheticState:
        transitioned = apply_action(state.game, action).state
        return SyntheticState(
            transitioned,
            state.action_count + 1,
            TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,)),
        )


def _first_legal_agents() -> dict[PlayerId, Agent]:
    return {
        player: ScriptedAgent((lambda decision: decision.legal_actions[0],) * 2)
        for player in PlayerId
    }


def _drive_random_batch(
    specs: tuple[GameSpec, ...], agent_seeds: dict[str, tuple[int, int]]
) -> PullGameRunner[SyntheticState]:
    runner = PullGameRunner(SyntheticTerminalEngine(), specs)
    agents = {
        game_id: {player: RandomAgent(seeds[index]) for index, player in enumerate(PlayerId)}
        for game_id, seeds in agent_seeds.items()
    }
    while len(runner.results()) != len(specs):
        pending = runner.pending()
        submissions = tuple(
            Submission(
                request.game_id,
                agents[request.game_id][request.decision.chooser].choose_action(request.decision),
            )
            for request in reversed(pending)
        )
        runner.submit(submissions)
    return runner


def test_single_game_runner_is_deterministic_and_records_setup_and_turn_paths() -> None:
    engine: RunnerEngine[SyntheticState] = SyntheticTerminalEngine()
    first = SingleGameRunner(engine).run(901, _first_legal_agents(), game_id="fixture")
    second = SingleGameRunner(engine).run(901, _first_legal_agents(), game_id="fixture")

    assert first.record == second.record
    assert first.state == second.state
    assert tuple(action.decision_id for action in first.record.actions) == (1, 2, 3)
    assert first.record.terminal.reason is TerminalReason.CARD_EFFECT
    assert first.record.final_state == first.record.actions[-1].resulting_state
    assert first.record.initial_state != first.record.final_state


def test_pull_runner_exposes_simultaneous_setup_and_accepts_partial_submissions() -> None:
    runner = PullGameRunner(
        SyntheticTerminalEngine(),
        (GameSpec("a", 902), GameSpec("b", 903)),
    )
    pending = runner.pending()
    assert tuple((item.game_id, item.decision.decision_id) for item in pending) == (
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    )

    runner.submit(Submission("a", pending[0].decision.legal_actions[0]))
    assert tuple((item.game_id, item.decision.decision_id) for item in runner.pending()) == (
        ("a", 2),
        ("b", 1),
        ("b", 2),
    )
    assert runner.state("a").action_count == 1
    assert runner.result("a") is None
    assert runner.results() == ()


def test_multi_game_records_equal_independent_sequential_runs() -> None:
    specs = (GameSpec("a", 904), GameSpec("b", 905), GameSpec("c", 906))
    seeds = {"a": (10, 11), "b": (20, 21), "c": (30, 31)}
    batched = _drive_random_batch(specs, seeds)

    sequential_records = []
    for spec in specs:
        agents = {
            player: RandomAgent(seeds[spec.game_id][index]) for index, player in enumerate(PlayerId)
        }
        sequential_records.append(
            SingleGameRunner(SyntheticTerminalEngine())
            .run(spec.setup_seed, agents, game_id=spec.game_id)
            .record
        )

    assert tuple(result.record for result in batched.results()) == tuple(sequential_records)
    assert batched.game_ids == ("a", "b", "c")


def test_submit_validates_entire_batch_before_mutating_games() -> None:
    runner = PullGameRunner(SyntheticTerminalEngine(), (GameSpec("a", 907),))
    pending = runner.pending()
    valid = Submission("a", pending[0].decision.legal_actions[0])
    illegal = Submission("a", DrawAction(999))
    before = runner.state("a")

    assert runner.submit(()) == ()
    illegal_current = Submission("a", DrawAction(pending[0].decision.decision_id))
    with pytest.raises(SubmissionError, match="illegal"):
        runner.submit(illegal_current)
    with pytest.raises(SubmissionError, match="not pending"):
        runner.submit((valid, illegal))
    assert runner.state("a") == before

    with pytest.raises(SubmissionError, match="duplicate submission"):
        runner.submit((valid, valid))
    with pytest.raises(UnknownGameError, match="unknown"):
        runner.submit(Submission("missing", valid.action))

    terminal_mid_batch = PullGameRunner(TerminalAfterOneEngine(), (GameSpec("terminal", 914),))
    both = terminal_mid_batch.pending()
    with pytest.raises(SubmissionError, match="ended before"):
        terminal_mid_batch.submit(
            tuple(Submission(item.game_id, item.decision.legal_actions[0]) for item in both)
        )
    assert terminal_mid_batch.state("terminal").action_count == 0


def test_runner_reports_duplicate_unknown_blocked_and_step_limit_states() -> None:
    runner = PullGameRunner(SyntheticTerminalEngine(), (GameSpec("a", 908),))
    with pytest.raises(DuplicateGameError):
        runner.add_game(GameSpec("a", 909))
    with pytest.raises(UnknownGameError):
        runner.state("missing")
    with pytest.raises(UnknownGameError):
        runner.result("missing")
    with pytest.raises(ValueError, match="game ID"):
        GameSpec("", 1)
    with pytest.raises(RunnerError, match="duplicate decision"):
        PullGameRunner(DuplicateDecisionEngine(), (GameSpec("dup", 910),)).pending()
    with pytest.raises(ValueError, match="positive"):
        SingleGameRunner(SyntheticTerminalEngine(), max_actions=0)
    with pytest.raises(StepLimitError):
        SingleGameRunner(SyntheticTerminalEngine(), max_actions=1).run(911, _first_legal_agents())
    with pytest.raises(RunnerError, match="no agent"):
        SingleGameRunner(SyntheticTerminalEngine()).run(911, {})

    initial_terminal = PullGameRunner(InitiallyTerminalEngine(), (GameSpec("done", 912),))
    assert initial_terminal.pending() == ()
    assert initial_terminal.result("done") is not None
    assert initial_terminal.results()[0].record.actions == ()


def test_the_real_engine_adapter_plays_a_complete_game_including_dogma() -> None:
    """WP5 gate: a Dogma action resolves to the next decision, so a runner is never blocked."""

    adapter = InnovationEngineAdapter()
    took_dogma = False
    completed = 0
    # Whether a specific seed ever reaches an implemented top card depends on the shuffle, so
    # this walks a small deterministic seed batch and requires at least one real dogma action.
    for seed in range(913, 921):
        game_id = f"integration-{seed}"
        runner = PullGameRunner(adapter, (GameSpec(game_id, seed),))
        setup = runner.pending()
        runner.submit(
            tuple(Submission(item.game_id, item.decision.legal_actions[0]) for item in setup)
        )
        for _ in range(600):
            if runner.result(game_id) is not None:
                break
            pending = runner.pending()
            assert pending, "a non-terminal game must always expose a decision"
            assert runner.blocked_game_ids() == ()
            request = pending[0]
            dogma = next(
                (
                    action
                    for action in request.decision.legal_actions
                    if isinstance(action, DogmaAction)
                ),
                None,
            )
            chosen = dogma or request.decision.legal_actions[0]
            took_dogma = took_dogma or dogma is not None
            runner.submit(Submission(request.game_id, chosen))
        result = runner.result(game_id)
        assert result is not None, f"seed {seed} must terminate within the step ceiling"
        assert result.record.terminal is not None
        completed += 1
    assert completed == 8
    assert took_dogma, "the batch must actually exercise a dogma action"


def test_a_single_game_runner_completes_a_real_game_with_agents() -> None:
    adapter = InnovationEngineAdapter()
    result = SingleGameRunner(adapter, max_actions=2000).run(
        914,
        {player: RandomAgent(seed=7 + index) for index, player in enumerate(PlayerId)},
    )
    assert result.record.terminal is not None
    assert result.record.actions
