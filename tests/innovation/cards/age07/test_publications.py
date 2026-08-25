"""PUBLICATIONS: arbitrary stack order and optional yellow/blue up-splay."""

from __future__ import annotations

from support import choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_rearranging_preserves_splay_until_the_second_effect_changes_it() -> None:
    state = (
        scenario(REGISTRY)
        .board(
            P1,
            Color.BLUE,
            ("tools", "calendar", "publications"),
            splay=SplayDirection.RIGHT,
        )
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    result = resolve_dogma(
        state,
        "publications",
        choose_color(Color.BLUE),
        choose_card("publications"),
        choose_card("tools"),
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    stack = result.state.player(P1).board.stack(Color.BLUE)
    assert stack.cards == (
        CardId("publications"),
        CardId("tools"),
        CardId("calendar"),
    )
    assert stack.splay is SplayDirection.UP


def test_only_existing_yellow_or_blue_stacks_are_splay_options() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("publications",))
        .board(P1, Color.RED, ("metalworking", "archery"))
        .board(P2, Color.RED, ("construction",))
        .build()
    )
    result = resolve_dogma(
        state,
        "publications",
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    splay_decision = result.decisions[-1]
    offered = {getattr(action, "color", None) for action in splay_decision.legal_actions}
    assert Color.BLUE in offered
    assert Color.YELLOW not in offered
    assert Color.RED not in offered
