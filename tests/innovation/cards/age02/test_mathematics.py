"""MATHEMATICS: optional return, value snapshot, atomic meld, sharing, and terminal draw."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("mathematics",))
        .board(P2, Color.RED, ("archery",))
    )


def test_empty_or_declined_return_does_nothing() -> None:
    empty = resolve_dogma(_solo().build(), "mathematics", registry=REGISTRY, programs=PROGRAMS)
    assert empty.decisions == ()
    state = _solo().hand(P1, ("tools",)).build()
    declined = resolve_dogma(state, "mathematics", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert declined.state.player(P1).hand == (CardId("tools"),)


def test_returning_a_two_draws_and_melds_a_three() -> None:
    state = _solo().hand(P1, ("currency",)).supply(3, ("alchemy",)).build()
    result = resolve_dogma(
        state,
        "mathematics",
        choose_card("currency"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert CardId("currency") in result.state.supply.pile(2)
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("alchemy")


def test_returning_a_ten_requests_eleven_and_ends_the_game() -> None:
    state = _solo().hand(P1, ("databases",)).build()
    result = resolve_dogma(
        state,
        "mathematics",
        choose_card("databases"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert CardId("databases") in result.state.supply.pile(10)


def test_a_shared_return_and_meld_earns_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("mathematics",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .hand(P1, ("currency",))
        .hand(P2, ("canal-building",))
        .supply(3, ("alchemy", "compass"))
        .supply(2, ("construction",))
        .build()
    )
    result = resolve_dogma(
        state,
        "mathematics",
        choose_card("canal-building"),
        choose_card("currency"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P2).board.stack(Color.BLUE).top == CardId("alchemy")
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("compass")
    assert len(result.state.player(P1).hand) == 1
    assert CardId("construction") in result.state.supply.pile(2)
