"""SATELLITES hand reset ordering and nested non-demand execution."""

from __future__ import annotations

from support import choose_card, scenario
from support.assertions import round_trip_state
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


def _run_ordinal(state: GameState, ordinal: int, *pickers: ChoicePicker) -> EffectResolution:
    context = EffectContext(P1, P1, P1, P1, SATELLITES, None, state.turn_number, 1)
    result = start_program_effect(
        state,
        PROGRAMS.program_for_card(SATELLITES).program_id,
        ordinal,
        context,
        PROGRAMS,
        REGISTRY,
    )
    pending = list(pickers)
    while result.status is EffectStatus.AWAIT_DECISION:
        assert result.decision is not None and pending
        state_at_choice = round_trip_state(result.state, REGISTRY)
        result = submit_effect_action(
            state_at_choice,
            pending.pop(0)(result.decision),
            PROGRAMS,
            REGISTRY,
        )
    assert not pending
    return result


def test_all_hand_cards_are_returned_in_the_chosen_same_age_order_before_three_draws() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, (SATELLITES,))
        .hand(P1, ("tools", "writing"))
        .supply(8, ("antibiotics", "corporations", "empiricism"))
        .build()
    )
    result = _run_ordinal(state, 1, choose_card("writing"))
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {
        CardId("antibiotics"),
        CardId("corporations"),
        CardId("empiricism"),
    }
    assert result.state.supply.pile(1)[-2:] == (CardId("writing"), CardId("tools"))


def test_melded_card_executes_its_non_demand_effects_in_a_nested_scope() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, (SATELLITES,))
        .hand(P1, ("ecology", "tools"))
        .supply(10, ("databases", "robotics"))
        .build()
    )
    result = _run_ordinal(
        state,
        3,
        choose_card("ecology"),
        choose_card("tools"),
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("ecology")
    assert set(result.state.player(P1).hand) == {CardId("databases"), CardId("robotics")}
    assert result.state.supply.pile(1)[-1] == CardId("tools")
    ecology_events = tuple(
        event for event in result.events if event.source_card_id == CardId("ecology")
    )
    assert ecology_events and all(event.nested and not event.demand for event in ecology_events)
