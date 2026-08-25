"""LIGHTING: bounded subset, distinct-value reward, and separate tuck order."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, finish, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _base() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("lighting",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_three_selected_cards_with_two_values_score_exactly_two_sevens() -> None:
    state = (
        _base()
        .hand(P1, ("canal-building", "tools", "writing"))
        .supply(7, ("bicycle", "combustion"))
        .build()
    )
    result = resolve_dogma(
        state,
        "lighting",
        choose_card("canal-building"),
        choose_card("tools"),
        choose_card("writing"),
        # The blue cards share a destination stack, so their tuck order is a later decision.
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    for card_id in (CardId("bicycle"), CardId("combustion")):
        reward_events = tuple(event for event in result.events if card_id in event.card_ids)
        assert len(reward_events) == 2
        assert reward_events[0].atomic_group_id is not None
        assert len({event.atomic_group_id for event in reward_events}) == 1
    assert set(result.state.player(P1).score_pile) == {CardId("bicycle"), CardId("combustion")}
    assert result.state.player(P1).board.stack(Color.BLUE).cards == (
        CardId("tools"),
        CardId("writing"),
    )
    selection_decisions = result.decisions[:3]
    assert all(decision.context is not None for decision in selection_decisions)
    assert result.decisions[3].context is not None
    assert result.decisions[3].context.selected_so_far == ()


def test_selecting_nothing_skips_tucks_and_rewards() -> None:
    state = _base().hand(P1, ("tools",)).supply(7, ("bicycle",)).build()
    result = resolve_dogma(
        state,
        "lighting",
        finish(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert not result.state.player(P1).score_pile
    assert result.qualifying_changes == 0
