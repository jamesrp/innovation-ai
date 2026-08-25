"""Small adapter boundary between runners and a deterministic game engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
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
    """Runner adapter over the Innovation engine's public transition API.

    Every state that is not terminal exposes at least one player decision, including a state
    paused mid-dogma, so the runner never has to treat a running game as blocked.
    """

    registry: CardRegistry = field(default_factory=load_card_registry)
    programs: EffectProgramRegistry = field(default_factory=load_effect_programs)

    def initial_state(self, seed: int, /) -> GameState:
        """Build the seeded Innovation setup state."""

        return build_setup_state(seed, self.registry)

    def pending_decisions(self, state: GameState, /) -> tuple[Decision, ...]:
        """Project the engine's current player decisions."""

        return current_decisions(state, self.registry, self.programs)

    def apply(self, state: GameState, action: SemanticAction, /) -> GameState:
        """Apply through the pure transition API."""

        return apply_action(state, action, self.registry, self.programs).state

    def terminal_result(self, state: GameState, /) -> TerminalResult | None:
        """Read the state's typed terminal result."""

        return state.terminal_result

    def fingerprint(self, state: GameState, /) -> str:
        """Hash the complete authoritative state deterministically."""

        return state_hash(state)
