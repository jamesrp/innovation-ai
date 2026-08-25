"""FEUDALISM: victim-owned hidden choice, if-you-do unsplay, and bounded color splay."""

from __future__ import annotations

from support import choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_the_victim_transfers_a_castle_card_and_unsplays_that_cards_color() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("feudalism",))
        .board(P1, Color.RED, ("engineering",))
        .board(P2, Color.RED, ("optics", "colonialism"), splay=SplayDirection.LEFT)
        .hand(P2, ("archery", "machinery"))
        .build()
    )
    result = resolve_dogma(
        state,
        "feudalism",
        choose_card("archery"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions[0].chooser is P2
    assert CardId("archery") in result.state.player(P1).hand
    assert result.state.player(P2).board.stack(Color.RED).splay is SplayDirection.NONE


def test_no_castle_in_hand_skips_the_if_you_do_unsplay() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("feudalism",))
        .board(P2, Color.RED, ("optics", "colonialism"), splay=SplayDirection.LEFT)
        .hand(P2, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        "feudalism",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).board.stack(Color.RED).splay is SplayDirection.LEFT


def test_effect_two_offers_only_present_yellow_or_purple_stacks() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("mysticism", "feudalism"))
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "feudalism",
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.PURPLE}
    assert result.state.player(P1).board.stack(Color.PURPLE).splay is SplayDirection.LEFT
