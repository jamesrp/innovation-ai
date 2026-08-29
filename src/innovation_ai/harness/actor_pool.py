"""Bounded game actor lifecycle built over :mod:`harness.runner` pull semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from innovation_ai.harness.engine import RunnerEngine
from innovation_ai.harness.records import GameResult, RunnerRecording
from innovation_ai.harness.runner import (
    GameSpec,
    PendingGameDecision,
    PullGameRunner,
    RunnerError,
    Submission,
)

type CompletionHandler[StateT] = Callable[[GameResult[StateT]], None]


class BoundedActorPool[StateT]:
    """Keep at most ``max_games_in_flight`` pull-runner games alive at once.

    Completed games are handed to ``on_complete`` and immediately retired from the underlying
    runner before replacement specs are added. The pool intentionally retains only counters, not
    completed states, records, or action lists.
    """

    def __init__(
        self,
        engine: RunnerEngine[StateT],
        game_specs: Iterable[GameSpec],
        *,
        max_games_in_flight: int,
        on_complete: CompletionHandler[StateT] | None = None,
        recording: RunnerRecording | None = None,
    ) -> None:
        if max_games_in_flight < 1:
            raise ValueError("max games in flight must be positive")
        self._runner = PullGameRunner(engine, recording=recording)
        self._remaining = iter(game_specs)
        self._max_games_in_flight = max_games_in_flight
        self._on_complete = on_complete
        self._seen_game_ids: set[str] = set()
        self._completed_count = 0
        self._exhausted = False
        self._refill()

    @property
    def runner(self) -> PullGameRunner[StateT]:
        """Return the live pull runner for read-only scheduling integration.

        Schedulers may inspect this runner to build a snapshot, but callers must
        submit returned actions through :meth:`submit` so completed games retire
        and vacant actor slots refill correctly.
        """

        return self._runner

    @property
    def max_games_in_flight(self) -> int:
        """Return the strict upper bound on retained live game state."""

        return self._max_games_in_flight

    @property
    def in_flight_count(self) -> int:
        """Return the number of active or terminal-not-yet-retired runner slots."""

        return len(self._runner.game_ids)

    @property
    def game_ids(self) -> tuple[str, ...]:
        """Return current live game IDs in deterministic runner insertion order."""

        return self._runner.game_ids

    @property
    def completed_count(self) -> int:
        """Return how many games have been retired without retaining their results."""

        return self._completed_count

    @property
    def exhausted(self) -> bool:
        """Whether no unstarted specs remain."""

        return self._exhausted

    @property
    def is_finished(self) -> bool:
        """Whether every supplied game has completed and been retired."""

        return self._exhausted and self.in_flight_count == 0

    def pending(self) -> tuple[PendingGameDecision, ...]:
        """Return all decisions from the currently bounded live set."""

        return self._runner.pending()

    def blocked_game_ids(self) -> tuple[str, ...]:
        """Forward blocked-engine diagnostics for current slots."""

        return self._runner.blocked_game_ids()

    def submit(
        self, submissions: Submission | Iterable[Submission]
    ) -> tuple[GameResult[StateT], ...]:
        """Submit actions, retire completed games, and refill vacant slots deterministically."""

        completed = self._runner.submit(submissions)
        self._retire(completed)
        self._refill()
        return completed

    def _retire(self, completed: tuple[GameResult[StateT], ...]) -> None:
        for result in completed:
            retired = self._runner.retire_game(result.record.game_id)
            if retired is not result:  # pragma: no cover - defensive runner contract assertion
                raise RunnerError("runner retired a different completed game result")
            self._completed_count += 1
            if self._on_complete is not None:
                self._on_complete(result)

    def _refill(self) -> None:
        while self.in_flight_count < self._max_games_in_flight and not self._exhausted:
            try:
                spec = next(self._remaining)
            except StopIteration:
                self._exhausted = True
                break
            if spec.game_id in self._seen_game_ids:
                raise RunnerError(f"actor pool received duplicate game ID: {spec.game_id}")
            self._seen_game_ids.add(spec.game_id)
            self._runner.add_game(spec)
            result = self._runner.result(spec.game_id)
            if result is not None:
                self._retire((result,))


ActorPool = BoundedActorPool
