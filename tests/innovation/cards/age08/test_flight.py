"""FLIGHT red-splay prerequisite and its two optional splay instructions."""

from __future__ import annotations

from support import choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_red_already_up_allows_any_present_color_to_splay_up() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("combustion", "flight"), splay=SplayDirection.UP)
        .board(P1, Color.BLUE, ("tools", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "flight",
        choose_color(Color.BLUE),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.BLUE, Color.RED}
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.UP


def test_red_not_up_skips_the_first_choice_but_the_second_can_splay_red() -> None:
    state = scenario(REGISTRY).board(P1, Color.RED, ("combustion", "flight")).build()
    result = resolve_dogma(
        state,
        "flight",
        choose_color(Color.RED),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.UP
