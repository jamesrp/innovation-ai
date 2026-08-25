"""MOBILITY two-highest selection, eligibility filters, and conditional draw."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_two_highest_eligible_tops_transfer_together_and_trigger_one_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("mobility",))
        .board(P2, Color.BLUE, ("tools", "genetics"))
        .board(P2, Color.GREEN, ("mass-media",))
        .board(P2, Color.PURPLE, ("empiricism",))
        .board(P2, Color.YELLOW, ("antibiotics",))
        .supply(8, ("socialism",))
        .build()
    )
    result = resolve_dogma(
        state,
        "mobility",
        choose_card("genetics"),
        choose_card("mass-media"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    offered = {
        action.card_id
        for action in result.decisions[1].legal_actions
        if isinstance(action, ChooseCardAction)
    }
    assert offered == {CardId("antibiotics"), CardId("empiricism"), CardId("mass-media")}
    assert set(result.state.player(P1).score_pile) == {
        CardId("genetics"),
        CardId("mass-media"),
    }
    assert result.state.player(P2).hand == (CardId("socialism"),)
    transfer_events = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "transfer"
    )
    assert len(transfer_events) == 1
    assert transfer_events[0].change is not None
    assert len(transfer_events[0].change.card_moves) == 2
