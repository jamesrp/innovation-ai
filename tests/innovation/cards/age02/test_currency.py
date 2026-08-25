"""CURRENCY: canonical subset, separate return order, distinct values, atomic rewards."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, finish, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.GREEN, ("currency",)).board(P2, Color.RED, ("archery",))
    )


def test_an_empty_hand_has_no_decision_or_reward() -> None:
    result = resolve_dogma(_solo().build(), "currency", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).score_pile


def test_selecting_no_cards_declines_the_effect() -> None:
    state = _solo().hand(P1, ("tools",)).build()
    result = resolve_dogma(state, "currency", finish(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert not result.state.player(P1).score_pile


def test_each_distinct_returned_value_draws_and_scores_one_two() -> None:
    state = (
        _solo()
        .hand(P1, ("agriculture", "alchemy", "tools"))
        .supply(2, ("canal-building", "construction"))
        .build()
    )
    result = resolve_dogma(
        state,
        "currency",
        choose_card("agriculture"),
        choose_card("alchemy"),
        choose_card("tools"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (
        CardId("canal-building"),
        CardId("construction"),
    )
    assert not result.state.player(P1).hand
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("tools")) < pile.index(CardId("agriculture"))


def test_same_value_returns_produce_only_one_reward() -> None:
    state = (
        _solo()
        .hand(P1, ("agriculture", "tools"))
        .supply(2, ("canal-building", "construction"))
        .build()
    )
    result = resolve_dogma(
        state,
        "currency",
        choose_card("agriculture"),
        choose_card("tools"),
        choose_card("agriculture"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("canal-building"),)
    assert CardId("construction") in result.state.supply.pile(2)


def test_multi_choice_execution_round_trips_every_replay_checkpoint() -> None:
    state = (
        _solo()
        .hand(P1, ("agriculture", "alchemy", "tools"))
        .supply(2, ("canal-building", "construction"))
        .build()
    )
    result = resolve_dogma(
        state,
        "currency",
        choose_card("agriculture"),
        choose_card("alchemy"),
        finish(),
        registry=REGISTRY,
        programs=PROGRAMS,
        verify_resume=True,
    )
    assert len(result.decisions) == 3
