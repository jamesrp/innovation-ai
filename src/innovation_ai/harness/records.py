"""Immutable runner records, deliberately separate from WP9 log schemas."""

from __future__ import annotations

from dataclasses import dataclass

from innovation_ai.innovation.actions import SemanticAction
from innovation_ai.innovation.state import TerminalResult
from innovation_ai.innovation.types import PlayerId


@dataclass(frozen=True, slots=True)
class RecordedAction:
    """One submitted semantic action and its resulting state fingerprint."""

    decision_id: int
    chooser: PlayerId
    action: SemanticAction
    resulting_state: str


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
