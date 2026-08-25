"""AGRICULTURE: optional return, snapshotted reward value, sharing, and terminal draw."""

from __future__ import annotations

from support import assert_conserved, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo():  # type: ignore[no-untyped-def]
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("agriculture",))
        .board(P2, Color.RED, ("archery",))
    )


def test_an_empty_hand_has_no_effect_or_decision() -> None:
    result = resolve_dogma(_solo().build(), "agriculture", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.qualifying_changes == 0


def test_the_optional_return_can_be_declined() -> None:
    state = _solo().hand(P1, ("writing",)).build()
    result = resolve_dogma(state, "agriculture", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("writing"),)
    assert not result.state.player(P1).score_pile


def test_returning_a_card_draws_and_scores_one_value_higher() -> None:
    state = _solo().hand(P1, ("canal-building",)).supply(3, ("alchemy",)).build()
    result = resolve_dogma(
        state,
        "agriculture",
        choose_card("canal-building"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)
    assert CardId("canal-building") in result.state.supply.pile(2)
    assert_conserved(result.state, REGISTRY)


def test_returning_a_ten_requests_eleven_and_ends_the_game() -> None:
    state = _solo().hand(P1, ("databases",)).build()
    result = resolve_dogma(
        state,
        "agriculture",
        choose_card("databases"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert CardId("databases") in result.state.supply.pile(10)
    assert not result.state.player(P1).score_pile


def test_a_shared_return_and_reward_grant_exactly_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("agriculture",))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("writing",))
        .hand(P2, ("tools",))
        .supply(2, ("canal-building", "construction"))
        .build()
    )
    result = resolve_dogma(
        state,
        "agriculture",
        choose_card("tools"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert len(result.state.player(P2).score_pile) == 1
    assert len(result.state.player(P1).score_pile) == 1
    assert len(result.state.player(P1).hand) == 1
