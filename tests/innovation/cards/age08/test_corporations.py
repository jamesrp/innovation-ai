"""CORPORATIONS demand filtering and both drawn-and-melded rewards."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_successful_demand_scores_the_factory_top_and_both_executors_meld_eights() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("corporations",))
        .board(P2, Color.YELLOW, ("skyscrapers",))
        .supply(8, ("flight", "socialism"))
        .build()
    )
    result = resolve_dogma(
        state,
        "corporations",
        choose_card("skyscrapers"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert len(result.decisions) == 1
    for card_id in (CardId("flight"), CardId("socialism")):
        meld_events = tuple(event for event in result.events if card_id in event.card_ids)
        assert len(meld_events) == 2
        assert meld_events[0].atomic_group_id is not None
        assert len({event.atomic_group_id for event in meld_events}) == 1
    assert result.state.player(P1).score_pile == (CardId("skyscrapers"),)
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("flight")
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("socialism")
