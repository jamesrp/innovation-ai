"""PERSPECTIVE: optional return, bulb quantity, partial scoring, and sharing."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("perspective",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_returning_a_card_scores_one_card_for_the_sources_two_bulbs() -> None:
    state = _solo().hand(P1, ("tools", "writing")).build()
    result = resolve_dogma(
        state,
        "perspective",
        choose_card("tools"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("writing"),)
    assert not result.state.player(P1).hand
    assert CardId("tools") in result.state.supply.pile(1)


def test_declining_the_return_skips_all_scoring() -> None:
    state = _solo().hand(P1, ("tools", "writing")).build()
    result = resolve_dogma(
        state,
        "perspective",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}
    assert not result.state.player(P1).score_pile


def test_mandatory_partial_execution_stops_when_the_hand_runs_out() -> None:
    state = _solo().board(P1, Color.PURPLE, ("education",)).hand(P1, ("tools", "writing")).build()
    result = resolve_dogma(
        state,
        "perspective",
        choose_card("tools"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    # Five bulbs request two scores, but only Writing remains after the return.
    assert result.state.player(P1).score_pile == (CardId("writing"),)
    assert len(result.decisions) == 2


def test_a_stronger_bulb_opponent_executes_first_and_causes_a_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("perspective",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .hand(P1, ("tools", "writing"))
        .hand(P2, ("sailing", "clothing"))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "perspective",
        choose_card("sailing"),
        choose_card("clothing"),
        choose_card("tools"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2, P1, P1)
    assert result.state.player(P2).score_pile == (CardId("clothing"),)
    assert set(result.state.player(P1).hand) == {CardId("anatomy")}
    assert result.state.player(P1).score_pile == (CardId("writing"),)
