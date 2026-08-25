"""GENETICS scores every card under its newly melded 10."""

from __future__ import annotations

from support import resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_all_preexisting_cards_in_the_destination_stack_are_scored() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("pottery", "tools", "genetics"))
        .supply(10, ("bioengineering",))
        .build()
    )
    result = resolve_dogma(
        state,
        "genetics",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.BLUE).cards == (CardId("bioengineering"),)
    assert set(result.state.player(P1).score_pile) == {
        CardId("pottery"),
        CardId("tools"),
        CardId("genetics"),
    }
    draw_meld = tuple(
        event for event in result.events if CardId("bioengineering") in event.card_ids
    )
    assert len(draw_meld) == 2
    assert len({event.atomic_group_id for event in draw_meld}) == 1
