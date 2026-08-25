"""QUANTUM THEORY exact-two gate and draw-then-draw-and-score sequence."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_exactly_two_returns_draw_one_ten_to_hand_then_score_the_next_ten() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("quantum-theory",))
        .hand(P1, ("tools", "writing"))
        .supply(10, ("databases", "robotics"))
        .build()
    )
    result = resolve_dogma(
        state,
        "quantum-theory",
        choose_card("tools"),
        choose_card("writing"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == (CardId("databases"),)
    assert result.state.player(P1).score_pile == (CardId("robotics"),)
    robotics_events = tuple(
        event for event in result.events if CardId("robotics") in event.card_ids
    )
    assert len(robotics_events) == 2
    assert robotics_events[0].atomic_group_id is not None
    assert len({event.atomic_group_id for event in robotics_events}) == 1
    databases_events = tuple(
        event for event in result.events if CardId("databases") in event.card_ids
    )
    assert len(databases_events) == 1
    assert databases_events[0].atomic_group_id != robotics_events[0].atomic_group_id
    assert result.changed_cards() == (
        CardId("writing"),
        CardId("tools"),
        CardId("databases"),
        CardId("robotics"),
        CardId("robotics"),
    )


def test_one_return_does_not_draw_any_tens() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("quantum-theory",))
        .hand(P1, ("tools",))
        .supply(10, ("databases", "robotics"))
        .build()
    )
    result = resolve_dogma(
        state,
        "quantum-theory",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert not result.state.player(P1).hand
    assert not result.state.player(P1).score_pile
    assert result.state.supply.pile(10)[:2] == (CardId("databases"), CardId("robotics"))
