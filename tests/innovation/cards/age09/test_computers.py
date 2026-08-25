"""COMPUTERS optional splay and nested age-10 execution."""

from __future__ import annotations

from support import choose_color, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_splay_choice_excludes_absent_green_and_drawn_a_i_executes_nested() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("computers",))
        .board(P1, Color.RED, ("archery",))
        .supply(10, ("a-i", "databases"))
        .build()
    )
    result = resolve_dogma(
        state,
        "computers",
        choose_color(Color.RED),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    color_actions = result.decisions[0].legal_actions
    assert {getattr(action, "color", None) for action in color_actions} == {Color.RED, None}
    # A singleton can legally be selected for a no-op splay.
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.NONE
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("a-i")
    assert CardId("databases") in result.state.player(P1).score_pile
    nested = tuple(event for event in result.events if event.source_card_id == CardId("a-i"))
    assert nested and all(event.nested for event in nested)
    draw_meld = tuple(
        event
        for event in result.events
        if event.source_card_id == CardId("computers") and CardId("a-i") in event.card_ids
    )
    assert len(draw_meld) == 2
    assert len({event.atomic_group_id for event in draw_meld}) == 1


def test_a_nested_a_i_win_immediately_unwinds_the_complete_production_chain() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("computers",))
        .board(P1, Color.RED, ("robotics",))
        .board(P2, Color.BLUE, ("software",))
        .supply(10, ("a-i", "databases"))
        .build()
    )
    result = resolve_dogma(
        state,
        "computers",
        choose_color(Color.RED),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None and result.terminal.winners == (P1,)
    assert result.state.pending_effects == ()
    assert result.state.effect_variables == ()
    assert result.state.revealed == ()
