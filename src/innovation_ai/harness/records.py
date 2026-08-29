"""Immutable runner records, deliberately separate from WP9 log schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from innovation_ai.innovation.actions import SemanticAction
from innovation_ai.innovation.state import TerminalResult
from innovation_ai.innovation.types import PlayerId


@dataclass(frozen=True, slots=True)
class RecordedAction:
    """One submitted semantic action and an optional resulting-state fingerprint.

    ``None`` is used by compact/self-play recording, which keeps only initial and final
    fingerprints. Existing runner construction keeps this field populated by default.
    """

    decision_id: int
    chooser: PlayerId
    action: SemanticAction
    resulting_state: str | None


@dataclass(frozen=True, slots=True)
class SemanticActionEvent:
    """A compact action event emitted after a submitted action commits.

    The event deliberately contains only semantic routing data and an optional transition
    fingerprint; it never exposes a decision observation or authoritative game state.
    """

    game_id: str
    setup_seed: int
    decision_id: int
    chooser: PlayerId
    action: SemanticAction
    resulting_state: str | None


class SemanticActionSink(Protocol):
    """Consume committed compact semantic action events."""

    def record_action(self, event: SemanticActionEvent, /) -> None:
        """Persist or aggregate one action event."""


@dataclass(frozen=True, slots=True)
class RunnerRecording:
    """Select audit-compatible or compact action recording for a pull runner."""

    retain_actions: bool = True
    transition_fingerprints: bool = True
    action_sink: SemanticActionSink | None = None

    def __post_init__(self) -> None:
        if self.transition_fingerprints and not self.retain_actions and self.action_sink is None:
            raise ValueError(
                "transition fingerprints require retained actions or a semantic action sink"
            )


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Deterministic lightweight execution record produced by a runner."""

    game_id: str
    setup_seed: int
    initial_state: str
    actions: tuple[RecordedAction, ...]
    terminal: TerminalResult
    final_state: str


@dataclass(frozen=True, slots=True)
class GameResult[StateT]:
    """A completed game's final opaque state and deterministic record."""

    state: StateT
    record: GameRecord
