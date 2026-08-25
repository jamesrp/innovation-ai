"""MINIATURIZATION return branch and distinct-score-value reward."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_returning_a_ten_draws_once_for_each_distinct_score_value() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("miniaturization",))
        .hand(P1, ("databases",))
        .score(P1, ("tools", "calendar", "alchemy"))
        .supply(10, ("a-i", "bioengineering", "globalization"))
        .build()
    )
    result = resolve_dogma(
        state,
        "miniaturization",
        choose_card("databases"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {
        CardId("a-i"),
        CardId("bioengineering"),
        CardId("globalization"),
    }
    assert CardId("databases") in result.state.supply.pile(10)


def test_returning_a_non_ten_draws_no_tens() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("miniaturization",))
        .hand(P1, ("tools",))
        .score(P1, ("calendar", "alchemy"))
        .supply(10, ("a-i",))
        .build()
    )
    result = resolve_dogma(
        state,
        "miniaturization",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P1).hand
    assert result.state.supply.pile(10)[0] == CardId("a-i")
