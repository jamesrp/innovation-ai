"""PAPER: constrained optional splay and a snapshotted draw per left-splayed color."""

from __future__ import annotations

from support import choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_splaying_green_left_is_counted_by_the_following_draw_effect() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel", "paper"))
        .board(P1, Color.BLUE, ("pottery", "alchemy"), splay=SplayDirection.LEFT)
        .board(P2, Color.RED, ("archery",))
        .supply(4, ("enterprise", "navigation"))
        .build()
    )
    result = resolve_dogma(
        state,
        "paper",
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.LEFT
    assert result.state.player(P1).hand == (CardId("enterprise"), CardId("navigation"))


def test_declining_the_splay_still_draws_for_colors_already_splayed_left() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("paper",))
        .board(P1, Color.BLUE, ("pottery", "alchemy"), splay=SplayDirection.LEFT)
        .board(P2, Color.RED, ("archery",))
        .supply(4, ("enterprise",))
        .build()
    )
    result = resolve_dogma(state, "paper", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("enterprise"),)


def test_only_present_green_or_blue_stacks_are_offered_even_when_singletons() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("paper",))
        .board(P2, Color.RED, ("archery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "paper",
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.GREEN}
    # Choosing a singleton is legal but cannot create a remembered splay direction.
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.NONE
