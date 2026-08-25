"""FERMENTING: visible-leaf colors, zero branch, quantity, and sharing bonus."""

from __future__ import annotations

from support import resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_each_color_with_a_visible_leaf_draws_one_two() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("fermenting",))
        .board(P1, Color.BLUE, ("calendar",))
        .board(P1, Color.GREEN, ("sailing",))
        .board(P2, Color.RED, ("archery",))
        .supply(2, ("canal-building", "construction", "currency"))
        .build()
    )
    result = resolve_dogma(state, "fermenting", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (
        CardId("canal-building"),
        CardId("construction"),
        CardId("currency"),
    )


def test_covered_unsplayed_leaf_cards_do_not_add_a_qualifying_color() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("agriculture", "fermenting"))
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.BLUE, ("writing",))
        .supply(2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(state, "fermenting", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("canal-building"),)


def test_a_shared_draw_grants_exactly_one_bonus_draw_to_the_activator() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("fermenting",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(2, ("canal-building", "construction", "currency"))
        .build()
    )
    result = resolve_dogma(state, "fermenting", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).hand == (CardId("canal-building"),)
    assert result.state.player(P1).hand == (CardId("construction"), CardId("currency"))
