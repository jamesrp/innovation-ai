"""Trusted expansion of sampled paid-turn candidates into player-safe afterstates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from innovation_ai.harness.policy import (
    CandidateRoute,
    ValuePosition,
    build_afterstate_value_position,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.protocol import apply_action
from innovation_ai.innovation.state import GameState, TerminalResult
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.determinizations import (
    InformationSetSpec,
    SampleVerificationError,
    verify_sampled_state,
)


class CandidateExpansionError(RuntimeError):
    """A sampled candidate could not be safely expanded through the normal engine."""


@dataclass(frozen=True, slots=True)
class TerminalCandidate:
    """One candidate whose exact result bypasses the value evaluator."""

    route: CandidateRoute
    utility: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.utility <= 1.0:
            raise ValueError("terminal candidate utility must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateExpansion:
    """Flattened nonterminal evaluator inputs plus exact terminal candidate values.

    ``routes`` and ``positions`` are deliberately parallel.  Terminal transitions are kept out
    of those two tuples so a model cannot accidentally be called for an already exact outcome.
    """

    routes: tuple[CandidateRoute, ...]
    positions: tuple[ValuePosition, ...]
    terminal_candidates: tuple[TerminalCandidate, ...]

    def __post_init__(self) -> None:
        if len(self.routes) != len(self.positions):
            raise ValueError("candidate routes and positions must remain parallel")
        all_routes = (*self.routes, *(item.route for item in self.terminal_candidates))
        keys = tuple((route.action, route.sample_index) for route in all_routes)
        if len(set(keys)) != len(keys):
            raise ValueError("a candidate action/sample pair cannot be expanded twice")

    @property
    def all_routes(self) -> tuple[CandidateRoute, ...]:
        """Return model and exact-terminal routes in deterministic expansion order."""

        return (*self.routes, *(item.route for item in self.terminal_candidates))


def terminal_utility(result: TerminalResult, viewer: PlayerId) -> float:
    """Return the exact Milestone-2 terminal utility from ``viewer``'s perspective."""

    if result.is_draw:
        return 0.5
    return 1.0 if viewer in result.winners else 0.0


class TrustedCandidateExpander:
    """Apply every semantic turn action to every independently sampled state.

    The expander intentionally has no method accepting a live original game state.  Its only
    authoritative inputs are states already reconstructed from an ``InformationSetSpec``; each is
    verified again before candidate expansion, making a real-state shortcut fail loudly.
    """

    def __init__(self, registry: CardRegistry | None = None) -> None:
        self._registry = registry or load_card_registry()

    def expand(
        self,
        spec: InformationSetSpec,
        samples: Sequence[GameState],
        *,
        game_id: str,
        evaluator_key: str,
    ) -> CandidateExpansion:
        """Expand all legal actions on every common-random-number sampled state."""

        if not game_id or not evaluator_key:
            raise ValueError("candidate expansion requires non-empty game and evaluator IDs")
        if not samples:
            raise CandidateExpansionError("candidate expansion requires at least one sampled state")

        routes: list[CandidateRoute] = []
        positions: list[ValuePosition] = []
        terminals: list[TerminalCandidate] = []
        for sample_index, sampled in enumerate(samples):
            try:
                verify_sampled_state(spec, sampled, self._registry)
            except SampleVerificationError as error:
                raise CandidateExpansionError(
                    f"sample {sample_index} is not consistent with its information set: {error}"
                ) from error
            for action in spec.legal_actions:
                route = CandidateRoute(
                    game_id,
                    spec.monotonic_ids.next_decision_id,
                    action,
                    sample_index,
                    evaluator_key,
                )
                transition = apply_action(sampled, action, self._registry)
                if transition.terminal is not None:
                    terminals.append(
                        TerminalCandidate(
                            route, terminal_utility(transition.terminal, spec.chooser)
                        )
                    )
                    continue
                if transition.decision is None:  # pragma: no cover - Transition contract guard
                    raise CandidateExpansionError(
                        "nonterminal candidate did not expose a next decision"
                    )
                routes.append(route)
                positions.append(
                    build_afterstate_value_position(
                        transition.state,
                        spec.chooser,
                        transition.decision,
                        self._registry,
                    )
                )
        return CandidateExpansion(tuple(routes), tuple(positions), tuple(terminals))


def expand_sampled_candidates(
    spec: InformationSetSpec,
    samples: Sequence[GameState],
    *,
    game_id: str,
    evaluator_key: str,
    registry: CardRegistry | None = None,
) -> CandidateExpansion:
    """Functional convenience wrapper around :class:`TrustedCandidateExpander`."""

    return TrustedCandidateExpander(registry).expand(
        spec,
        samples,
        game_id=game_id,
        evaluator_key=evaluator_key,
    )


__all__ = [
    "CandidateExpansion",
    "CandidateExpansionError",
    "TerminalCandidate",
    "TrustedCandidateExpander",
    "expand_sampled_candidates",
    "terminal_utility",
]
