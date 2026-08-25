"""THE INTERNET splay, score, and fixed clock-quantity melds."""

from __future__ import annotations

from support import choose_color, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_clock_quantity_is_snapshotted_before_new_clock_cards_are_melded() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("the-internet",))
        .board(P1, Color.GREEN, ("sailing", "clothing"))
        .supply(10, ("globalization", "software", "databases"))
        .build()
    )
    result = resolve_dogma(
        state,
        "the-internet",
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.UP
    assert result.state.player(P1).score_pile == (CardId("globalization"),)
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("software")
    # The Internet began effect three with two clocks, so Software's three new clocks do not
    # increase this execution from one meld to a second meld.
    assert result.state.supply.pile(10)[0] == CardId("databases")
