"""ANTIBIOTICS bounded returns and distinct-value rewards."""

from __future__ import annotations

from support import choose_card, finish, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_three_returns_with_two_values_draw_exactly_four_eights() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("antibiotics",))
        .hand(P1, ("calendar", "construction", "tools"))
        .supply(8, ("corporations", "empiricism", "flight", "mass-media"))
        .build()
    )
    result = resolve_dogma(
        state,
        "antibiotics",
        choose_card("calendar"),
        choose_card("construction"),
        choose_card("tools"),
        choose_card("tools"),
        choose_card("construction"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {
        CardId("corporations"),
        CardId("empiricism"),
        CardId("flight"),
        CardId("mass-media"),
    }
    assert result.state.supply.pile(1)[-1] == CardId("tools")
    assert result.state.supply.pile(2)[-2:] == (
        CardId("construction"),
        CardId("calendar"),
    )


def test_finishing_without_returns_draws_nothing() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("antibiotics",))
        .hand(P1, ("tools",))
        .supply(8, ("corporations",))
        .build()
    )
    result = resolve_dogma(
        state,
        "antibiotics",
        finish(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("tools"),)
