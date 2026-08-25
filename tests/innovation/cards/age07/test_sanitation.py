"""SANITATION: two highest versus one lowest with hidden-zone tie ownership."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_each_hidden_zone_owner_disambiguates_its_own_ties() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("sanitation",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P1, ("canal-building", "construction", "alchemy"))
        .hand(P2, ("banking", "chemistry", "coal", "tools"))
        .build()
    )
    result = resolve_dogma(
        state,
        "sanitation",
        choose_card("chemistry"),
        choose_card("coal"),
        choose_card("construction"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2, P1)
    first_offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert first_offered == {CardId("banking"), CardId("chemistry"), CardId("coal")}
    lowest_offered = {
        action.card_id for action in result.decisions[2].legal_actions if hasattr(action, "card_id")
    }
    assert lowest_offered == {CardId("canal-building"), CardId("construction")}
    assert set(result.state.player(P1).hand) == {
        CardId("canal-building"),
        CardId("alchemy"),
        CardId("chemistry"),
        CardId("coal"),
    }
    assert set(result.state.player(P2).hand) == {
        CardId("banking"),
        CardId("tools"),
        CardId("construction"),
    }
    exchange_events = tuple(event for event in result.events if event.change is not None)
    assert len(exchange_events) == 1
    assert exchange_events[0].change is not None
    assert exchange_events[0].change.kind.value == "exchange"
    assert len(exchange_events[0].change.card_moves) == 3


def test_partial_exchange_uses_every_available_side() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("sanitation",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P1, ("tools",))
        .hand(P2, ("banking",))
        .build()
    )
    result = resolve_dogma(state, "sanitation", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("banking"),)
    assert result.state.player(P2).hand == (CardId("tools"),)
