"""CALENDAR: comparison branches, two draws, opponent-first sharing, and bonus draw."""

from __future__ import annotations

from support import ScenarioBuilder, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.BLUE, ("calendar",)).board(P2, Color.RED, ("archery",))
    )


def test_equal_hand_and_score_counts_do_not_draw() -> None:
    result = resolve_dogma(_solo().build(), "calendar", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).hand


def test_more_scored_cards_than_hand_cards_draws_two_threes() -> None:
    state = _solo().score(P1, ("tools",)).supply(3, ("alchemy", "compass")).build()
    result = resolve_dogma(state, "calendar", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("alchemy"), CardId("compass"))


def test_a_sharing_opponent_executes_first_and_earns_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("calendar",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .score(P1, ("tools",))
        .score(P2, ("writing",))
        .supply(3, ("alchemy", "compass", "education", "engineering"))
        .supply(2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(state, "calendar", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).hand == (CardId("alchemy"), CardId("compass"))
    assert set(result.state.player(P1).hand) == {
        CardId("education"),
        CardId("engineering"),
        CardId("canal-building"),
    }
