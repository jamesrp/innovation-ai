"""ATOMIC THEORY: optional blue splay and atomic age-7 meld."""

from __future__ import annotations

from support import choose_color, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_blue_can_splay_right_before_the_seven_is_drawn_and_melded() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools", "atomic-theory"))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(7, ("bicycle",))
        .build()
    )
    result = resolve_dogma(
        state,
        "atomic-theory",
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.RIGHT
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("bicycle")
    draw_meld_events = tuple(event for event in result.events if event.change is not None)[-2:]
    assert len({event.atomic_group_id for event in draw_meld_events}) == 1


def test_declining_the_splay_does_not_skip_the_mandatory_meld() -> None:
    from support import decline

    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools", "atomic-theory"))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(7, ("combustion",))
        .build()
    )
    result = resolve_dogma(state, "atomic-theory", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.NONE
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("combustion")
