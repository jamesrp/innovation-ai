"""EMANCIPATION: victim-owned hand transfer, conditional draw, and optional splay."""

from __future__ import annotations

from support import choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_victim_selects_the_private_transfer_and_draws_only_after_it_moves() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("emancipation",))
        .board(P2, Color.BLUE, ("tools",))
        .hand(P2, ("writing", "sailing"))
        .supply(6, ("classification",))
        .build()
    )
    result = resolve_dogma(
        state,
        "emancipation",
        choose_card("writing"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P1).score_pile == (CardId("writing"),)
    assert set(result.state.player(P2).hand) == {CardId("sailing"), CardId("classification")}


def test_empty_victim_hand_skips_the_transfer_and_conditional_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("emancipation",))
        .board(P2, Color.BLUE, ("tools",))
        .supply(6, ("classification",))
        .build()
    )
    result = resolve_dogma(state, "emancipation", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert tuple(decision.chooser for decision in result.decisions) == (P1,)
    assert not result.state.player(P1).score_pile
    assert not result.state.player(P2).hand
