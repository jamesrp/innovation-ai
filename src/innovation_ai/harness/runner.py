"""Single-game and pull-based multi-game execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from innovation_ai.agents.base import Agent
from innovation_ai.harness.engine import RunnerEngine
from innovation_ai.harness.records import (
    GameRecord,
    GameResult,
    RecordedAction,
    RunnerRecording,
    SemanticActionEvent,
)
from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.types import PlayerId


class RunnerError(RuntimeError):
    """Base class for runner protocol failures."""


class DuplicateGameError(RunnerError):
    """A game ID was added more than once."""


class UnknownGameError(RunnerError):
    """A submission or lookup referenced an unknown game."""


class SubmissionError(RunnerError):
    """A submission did not answer a currently pending decision."""


class GameBlockedError(RunnerError):
    """A non-terminal game has no player decision the runner can submit."""


class StepLimitError(RunnerError):
    """A single-game run exceeded its defensive action ceiling."""


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Identity and deterministic setup seed for one independent game."""

    game_id: str
    setup_seed: int

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game ID cannot be empty")


@dataclass(frozen=True, slots=True)
class PendingGameDecision:
    """A decision paired with the game to which its answer must be submitted."""

    game_id: str
    decision: Decision


@dataclass(frozen=True, slots=True)
class Submission:
    """One semantic action routed to an independent game."""

    game_id: str
    action: SemanticAction


@dataclass(slots=True)
class _RunningGame[StateT]:
    spec: GameSpec
    state: StateT
    initial_state: str
    actions: list[RecordedAction]
    result: GameResult[StateT] | None = None


class PullGameRunner[StateT]:
    """Collect decisions from many games and accept externally selected actions.

    ``pending()`` is a pure pull operation. ``submit()`` accepts any subset of that snapshot,
    validates the full batch before committing it, and applies submissions in deterministic game
    and decision order rather than caller order. This keeps the engine independent from policies
    and leaves a direct batching seam for a future external model process.
    """

    def __init__(
        self,
        engine: RunnerEngine[StateT],
        games: Iterable[GameSpec] = (),
        *,
        recording: RunnerRecording | None = None,
    ) -> None:
        self._engine = engine
        self._recording = recording or RunnerRecording()
        self._games: dict[str, _RunningGame[StateT]] = {}
        for spec in games:
            self.add_game(spec)

    @property
    def game_ids(self) -> tuple[str, ...]:
        """Return game IDs in stable insertion order."""

        return tuple(self._games)

    def add_game(self, spec: GameSpec) -> None:
        """Add a seeded game; immediately retain an already-terminal setup if supported."""

        if spec.game_id in self._games:
            raise DuplicateGameError(f"duplicate game ID: {spec.game_id}")
        state = self._engine.initial_state(spec.setup_seed)
        initial = self._engine.fingerprint(state)
        game = _RunningGame(spec, state, initial, [])
        terminal = self._engine.terminal_result(state)
        if terminal is not None:
            record = GameRecord(
                spec.game_id,
                spec.setup_seed,
                initial,
                (),
                terminal,
                initial,
            )
            game.result = GameResult(state, record)
        self._games[spec.game_id] = game

    def pending(self) -> tuple[PendingGameDecision, ...]:
        """Return all current player decisions in game insertion and engine order."""

        requests: list[PendingGameDecision] = []
        for game_id, game in self._games.items():
            if game.result is not None:
                continue
            decisions = self._engine.pending_decisions(game.state)
            if len({decision.decision_id for decision in decisions}) != len(decisions):
                raise RunnerError(f"engine returned duplicate decision IDs for game {game_id}")
            requests.extend(PendingGameDecision(game_id, decision) for decision in decisions)
        return tuple(requests)

    def blocked_game_ids(self) -> tuple[str, ...]:
        """Return non-terminal games with no player-facing decision."""

        return tuple(
            game_id
            for game_id, game in self._games.items()
            if game.result is None and not self._engine.pending_decisions(game.state)
        )

    def submit(
        self, submissions: Submission | Iterable[Submission]
    ) -> tuple[GameResult[StateT], ...]:
        """Apply a validated singular or batched subset of the latest pending decisions."""

        submitted = (submissions,) if isinstance(submissions, Submission) else tuple(submissions)
        if not submitted:
            return ()

        pending = self.pending()
        pending_by_key = {
            (request.game_id, request.decision.decision_id): request for request in pending
        }
        selected: dict[tuple[str, int], tuple[Submission, Decision]] = {}
        for submission in submitted:
            if submission.game_id not in self._games:
                raise UnknownGameError(f"unknown game ID: {submission.game_id}")
            key = (submission.game_id, submission.action.decision_id)
            if key in selected:
                raise SubmissionError(f"duplicate submission for game {key[0]} decision {key[1]}")
            request = pending_by_key.get(key)
            if request is None:
                raise SubmissionError(f"decision {key[1]} is not pending for game {key[0]}")
            if submission.action not in request.decision.legal_actions:
                raise SubmissionError(
                    f"action {submission.action.kind.value} is illegal for game {key[0]} "
                    f"decision {key[1]}"
                )
            selected[key] = (submission, request.decision)

        working: dict[str, tuple[StateT, list[RecordedAction], GameResult[StateT] | None]] = {
            game_id: (game.state, list(game.actions), game.result)
            for game_id, game in self._games.items()
        }
        newly_completed: list[GameResult[StateT]] = []
        emitted_events: list[SemanticActionEvent] = []
        for request in pending:
            key = (request.game_id, request.decision.decision_id)
            chosen = selected.get(key)
            if chosen is None:
                continue
            submission, decision = chosen
            state, records, result = working[request.game_id]
            if result is not None:
                raise SubmissionError(f"game {request.game_id} ended before its submitted action")
            state = self._engine.apply(state, submission.action)
            fingerprint = (
                self._engine.fingerprint(state) if self._recording.transition_fingerprints else None
            )
            event = SemanticActionEvent(
                request.game_id,
                self._games[request.game_id].spec.setup_seed,
                decision.decision_id,
                decision.chooser,
                submission.action,
                fingerprint,
            )
            if self._recording.retain_actions:
                records.append(
                    RecordedAction(
                        decision.decision_id,
                        decision.chooser,
                        submission.action,
                        fingerprint,
                    )
                )
            emitted_events.append(event)
            terminal = self._engine.terminal_result(state)
            if terminal is not None:
                game = self._games[request.game_id]
                final_fingerprint = fingerprint or self._engine.fingerprint(state)
                record = GameRecord(
                    request.game_id,
                    game.spec.setup_seed,
                    game.initial_state,
                    tuple(records),
                    terminal,
                    final_fingerprint,
                )
                result = GameResult(state, record)
                newly_completed.append(result)
            working[request.game_id] = (state, records, result)

        for game_id, (state, records, result) in working.items():
            game = self._games[game_id]
            game.state = state
            game.actions = records
            game.result = result
        if self._recording.action_sink is not None:
            for event in emitted_events:
                self._recording.action_sink.record_action(event)
        return tuple(newly_completed)

    def retire_game(self, game_id: str) -> GameResult[StateT]:
        """Remove and return a completed game so long-running actors release its state/actions."""

        try:
            game = self._games[game_id]
        except KeyError as error:
            raise UnknownGameError(f"unknown game ID: {game_id}") from error
        if game.result is None:
            raise RunnerError(f"game {game_id} cannot retire before reaching a terminal result")
        del self._games[game_id]
        return game.result

    def state(self, game_id: str) -> StateT:
        """Return the current opaque state for integration and diagnostics."""

        try:
            return self._games[game_id].state
        except KeyError as error:
            raise UnknownGameError(f"unknown game ID: {game_id}") from error

    def result(self, game_id: str) -> GameResult[StateT] | None:
        """Return a completed result or ``None`` while the game is active."""

        try:
            return self._games[game_id].result
        except KeyError as error:
            raise UnknownGameError(f"unknown game ID: {game_id}") from error

    def results(self) -> tuple[GameResult[StateT], ...]:
        """Return completed results in game insertion order."""

        return tuple(game.result for game in self._games.values() if game.result is not None)


class SingleGameRunner[StateT]:
    """Drive one game by dispatching each decision to its chooser's agent."""

    def __init__(self, engine: RunnerEngine[StateT], *, max_actions: int = 10_000) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._engine = engine
        self._max_actions = max_actions

    def run(
        self,
        setup_seed: int,
        agents: Mapping[PlayerId, Agent],
        *,
        game_id: str = "game-0",
    ) -> GameResult[StateT]:
        """Run until terminal, blocked engine integration, or the action ceiling."""

        runner = PullGameRunner(self._engine, (GameSpec(game_id, setup_seed),))
        result = runner.result(game_id)
        actions = 0
        while result is None:
            pending = runner.pending()
            if not pending:
                raise GameBlockedError(
                    f"game {game_id} is non-terminal with no pending player decision"
                )
            if actions + len(pending) > self._max_actions:
                raise StepLimitError(f"game {game_id} exceeded action ceiling {self._max_actions}")
            submissions: list[Submission] = []
            for request in pending:
                try:
                    agent = agents[request.decision.chooser]
                except KeyError as error:
                    raise RunnerError(
                        f"no agent configured for {request.decision.chooser.value}"
                    ) from error
                submissions.append(
                    Submission(request.game_id, agent.choose_action(request.decision))
                )
            runner.submit(submissions)
            actions += len(submissions)
            result = runner.result(game_id)
        return result


MultiGameRunner = PullGameRunner
