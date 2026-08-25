"""Small adapter boundary between runners and a deterministic game engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import (
    GameState,
    TerminalResult,
    build_setup_state,
    state_hash,
)


class RunnerEngine[StateT](Protocol):
    """Opaque immutable-state operations required by game runners."""

    def initial_state(self, seed: int, /) -> StateT:
        """Construct one deterministic game from an explicit setup seed."""

    def pending_decisions(self, state: StateT, /) -> tuple[Decision, ...]:
        """Return all player decisions currently waiting for submissions."""

    def apply(self, state: StateT, action: SemanticAction, /) -> StateT:
        """Apply one legal action and return the resulting state."""

    def terminal_result(self, state: StateT, /) -> TerminalResult | None:
        """Return the terminal result, if the game has ended."""

    def fingerprint(self, state: StateT, /) -> str:
        """Return a deterministic identifier for the full authoritative state."""


@dataclass(frozen=True, slots=True)
class InnovationEngineAdapter:
    """Runner adapter over the current frozen Innovation engine APIs.

    Freeze A hands a selected Dogma action to later effect work by installing a pending frame.
    Until that resolver lands, such a state honestly has no player decision and is reported by
    runners as blocked rather than treating dogma as a no-op.
    """

    registry: CardRegistry = field(default_factory=load_card_registry)

    def initial_state(self, seed: int, /) -> GameState:
        """Build the seeded Innovation setup state."""

        return build_setup_state(seed, self.registry)

    def pending_decisions(self, state: GameState, /) -> tuple[Decision, ...]:
        """Project the engine's current player decisions."""

        return current_decisions(state, self.registry)

    def apply(self, state: GameState, action: SemanticAction, /) -> GameState:
        """Apply through the frozen pure transition API."""

        return apply_action(state, action, self.registry).state

    def terminal_result(self, state: GameState, /) -> TerminalResult | None:
        """Read the state's typed terminal result."""

        return state.terminal_result

    def fingerprint(self, state: GameState, /) -> str:
        """Hash the complete authoritative state deterministically."""

        return state_hash(state)
