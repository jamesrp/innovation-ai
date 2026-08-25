"""MAPMAKING: victim-owned score choice, demand history, immunity, and atomic reward."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.GREEN, ("mapmaking",)).board(P2, Color.RED, ("archery",))
    )


def test_no_value_one_in_the_victim_score_means_no_transfer_or_reward() -> None:
    state = _vulnerable().score(P2, ("canal-building",)).build()
    result = resolve_dogma(state, "mapmaking", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).score_pile
    assert result.state.player(P2).score_pile == (CardId("canal-building"),)


def test_the_victim_chooses_a_tied_one_then_the_activator_draws_and_scores() -> None:
    state = _vulnerable().score(P2, ("tools", "writing")).supply(1, ("agriculture",)).build()
    result = resolve_dogma(
        state,
        "mapmaking",
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2,)
    assert result.state.player(P2).score_pile == (CardId("tools"),)
    assert set(result.state.player(P1).score_pile) == {
        CardId("writing"),
        CardId("agriculture"),
    }
    reward_events = tuple(
        event for event in result.events if CardId("agriculture") in event.card_ids
    )
    assert len(reward_events) == 2
    assert reward_events[0].atomic_group_id == reward_events[1].atomic_group_id


def test_equal_crowns_skip_the_demand_and_the_conditional_reward() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("mapmaking",))
        .board(P2, Color.YELLOW, ("canal-building",))
        .score(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "mapmaking", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).score_pile == (CardId("tools"),)
    assert not result.state.player(P1).score_pile
