"""RAILROAD: owner-chosen bulk return order and right-to-up splay."""

from __future__ import annotations

from support import choose_card, choose_color, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_hand_returns_are_bulk_ordered_before_three_sixes_are_drawn() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("railroad",))
        .board(
            P1,
            Color.RED,
            ("metalworking", "construction"),
            splay=SplayDirection.RIGHT,
        )
        .board(
            P1,
            Color.BLUE,
            ("tools", "calendar"),
            splay=SplayDirection.LEFT,
        )
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("sailing", "writing", "canal-building"))
        .supply(6, ("machine-tools", "metric-system", "vaccination"))
        .build()
    )
    result = resolve_dogma(
        state,
        "railroad",
        choose_card("writing"),
        choose_color(Color.RED),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("writing")) < pile.index(CardId("sailing"))
    assert result.state.player(P1).hand == (
        CardId("machine-tools"),
        CardId("metric-system"),
        CardId("vaccination"),
    )
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.UP
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.LEFT


def test_only_tops_of_currently_right_splayed_stacks_are_offered() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("railroad",))
        .board(
            P1,
            Color.RED,
            ("metalworking", "construction"),
            splay=SplayDirection.RIGHT,
        )
        .board(
            P1,
            Color.BLUE,
            ("tools", "calendar"),
            splay=SplayDirection.LEFT,
        )
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "railroad",
        choose_color(Color.RED),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.RED}
