"""Shared assertions for effect, dogma, and card suites."""

from __future__ import annotations

from collections.abc import Sequence

from innovation_ai.innovation.actions import SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.invariants import (
    assert_card_conservation,
    assert_observation_leak_resistance,
    assert_state_properties,
    assert_unique_card_locations,
)
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.serialization import dumps_state, loads_state
from innovation_ai.innovation.state import GameState, state_hash
from innovation_ai.innovation.types import PlayerId


def round_trip_state(state: GameState, registry: CardRegistry | None = None) -> GameState:
    """Serialize and restore a state, requiring an identical hash.

    This is the single place the versioned state schema is exercised from card tests, so a new
    state field that was not added to the decoder fails here rather than silently vanishing.
    """

    registry = registry or load_card_registry()
    restored = loads_state(dumps_state(state), registry)
    if state_hash(restored) != state_hash(state):
        raise AssertionError("state did not round-trip to an identical hash")
    if restored != state:
        raise AssertionError("state round-trip produced an unequal state")
    return restored


def assert_resumes_identically(
    state: GameState,
    actions: Sequence[SemanticAction],
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> GameState:
    """Apply ``actions``, serializing and restoring at every decision boundary.

    Each action is applied twice: once from the live state and once from a state that was written
    to JSON and read back. Both must produce the same hash, which proves a paused position carries
    everything the engine needs to continue.
    """

    registry = registry or load_card_registry()
    resolved_programs = programs or load_effect_programs()
    live = state
    for index, action in enumerate(actions):
        restored = round_trip_state(live, registry)
        decisions = current_decisions(live, registry, resolved_programs)
        if not decisions:
            raise AssertionError(f"action {index} has no pending decision")
        if action not in decisions[0].legal_actions:
            raise AssertionError(
                f"action {index} ({action.kind.value}) is not legal for "
                f"decision {decisions[0].decision_id}"
            )
        direct = apply_action(live, action, registry, resolved_programs)
        resumed = apply_action(restored, action, registry, resolved_programs)
        if state_hash(direct.state) != state_hash(resumed.state):
            raise AssertionError(f"action {index} diverged after a serialize/restore cycle")
        assert_state_properties(direct.state, registry, resolved_programs)
        live = direct.state
    return live


def assert_conserved(state: GameState, registry: CardRegistry | None = None) -> None:
    """Require all 105 cards to be present exactly once in exactly one location."""

    registry = registry or load_card_registry()
    assert_unique_card_locations(state)
    assert_card_conservation(state, registry)


def assert_no_leak(
    first: GameState,
    second: GameState,
    viewer: PlayerId,
    registry: CardRegistry | None = None,
) -> None:
    """Require two hidden-equivalent positions to be indistinguishable to ``viewer``."""

    assert_observation_leak_resistance(first, second, viewer, registry or load_card_registry())
