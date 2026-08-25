"""EXPLOSIVES: three current-highest transfers, hidden ties, and the empty-hand draw."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _base() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.RED, ("explosives",)).board(P2, Color.BLUE, ("pottery",))
    )


def test_the_victim_chooses_tied_highest_cards_before_lower_values() -> None:
    state = _base().hand(P2, ("banking", "chemistry", "coal", "construction")).build()
    result = resolve_dogma(
        state,
        "explosives",
        choose_card("chemistry"),
        choose_card("coal"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    first_offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert first_offered == {CardId("banking"), CardId("chemistry"), CardId("coal")}
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2)
    assert set(result.state.player(P1).hand) == {
        CardId("banking"),
        CardId("chemistry"),
        CardId("coal"),
    }
    assert result.state.player(P2).hand == (CardId("construction"),)
    transfer_events = tuple(event for event in result.events if event.change is not None)
    assert len(transfer_events) == 1
    assert len(transfer_events[0].change.card_moves) == 3  # type: ignore[union-attr]


def test_fewer_than_three_cards_transfer_partially_then_an_empty_victim_draws_a_seven() -> None:
    state = _base().hand(P2, ("banking", "construction")).supply(7, ("bicycle",)).build()
    result = resolve_dogma(state, "explosives", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert set(result.state.player(P1).hand) == {CardId("banking"), CardId("construction")}
    assert result.state.player(P2).hand == (CardId("bicycle"),)


def test_an_initially_empty_hand_does_not_trigger_the_conditional_draw() -> None:
    state = _base().supply(7, ("bicycle",)).build()
    result = resolve_dogma(state, "explosives", registry=REGISTRY, programs=PROGRAMS)
    assert not result.state.player(P2).hand
