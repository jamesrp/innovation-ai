"""METRIC SYSTEM: right-splayed-green guard and any-present-colour choice."""

from __future__ import annotations

from support import choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_right_splayed_green_allows_any_present_colour_including_a_singleton() -> None:
    state = (
        scenario(REGISTRY)
        .board(
            P1,
            Color.GREEN,
            ("clothing", "metric-system"),
            splay=SplayDirection.RIGHT,
        )
        .board(P1, Color.BLUE, ("tools",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        "metric-system",
        choose_color(Color.BLUE),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.BLUE, Color.GREEN}
    # Choosing a singleton is legal but remains an unsplayed no-op (decision 15).
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.NONE


def test_without_right_splayed_green_only_the_second_instruction_is_offered() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("metric-system",))
        .board(P1, Color.BLUE, ("tools",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "metric-system", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert len(result.decisions) == 1
    offered = {
        action.color
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.GREEN}
