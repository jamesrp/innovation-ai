"""Demand-specific causal results stay isolated across production nested execution."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario
from support.scenario import ChoicePicker

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectResolution,
    EffectStatus,
    load_effect_programs,
    start_program_effect,
    submit_effect_action,
)
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()
SATELLITES = CardId("satellites")


def _run_satellites_nested(state: GameState, *pickers: ChoicePicker) -> EffectResolution:
    context = EffectContext(P1, P1, P1, P1, SATELLITES, None, state.turn_number, 1)
    result = start_program_effect(
        state,
        PROGRAMS.program_for_card(SATELLITES).program_id,
        3,
        context,
        PROGRAMS,
        REGISTRY,
    )
    pending = list(pickers)
    while result.status is EffectStatus.AWAIT_DECISION:
        assert result.decision is not None and pending
        result = submit_effect_action(
            result.state,
            pending.pop(0)(result.decision),
            PROGRAMS,
            REGISTRY,
        )
    assert not pending
    return result


def test_nested_pirate_code_ignores_the_outer_meld_change_when_its_demand_is_skipped() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, (SATELLITES,))
        .hand(P1, ("the-pirate-code",))
        .build()
    )
    result = _run_satellites_nested(state, choose_card("the-pirate-code"))

    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("the-pirate-code")
    assert not result.state.player(P1).score_pile


def test_nested_vaccination_ignores_the_outer_meld_change_when_its_demand_is_skipped() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, (SATELLITES,))
        .hand(P1, ("vaccination",))
        .supply(7, ("bicycle",))
        .build()
    )
    result = _run_satellites_nested(state, choose_card("vaccination"))

    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("vaccination")
    assert result.state.supply.pile(7)[0] == CardId("bicycle")


def test_nested_oars_draws_after_its_skipped_demand_despite_outer_satellites_changes() -> None:
    removed: list[CardId] = []
    for age in (8, 9):
        candidates = sorted(
            (card.id for card in REGISTRY.cards if card.age == age and card.id != SATELLITES),
            key=str,
        )
        # One unplaced card becomes the hidden normal achievement; all other cards leave the
        # supply, forcing every requested 8 upward to the pinned age-10 cards.
        removed.extend(candidates[1:])

    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, (SATELLITES,))
        .board(P1, Color.RED, ("oars",))
        .removed(removed)
        .supply(10, ("self-service", "databases", "robotics"))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        SATELLITES,
        choose_card("self-service"),
        choose_card("oars"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {
        CardId("databases"),
        CardId("robotics"),
        CardId("agriculture"),
    }
