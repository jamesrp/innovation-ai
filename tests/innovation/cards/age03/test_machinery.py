"""MACHINERY: mandatory atomic exchanges, empty sides, scoring, and optional splay."""

from __future__ import annotations

from support import ScenarioBuilder, choose_branch, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("machinery",))
        .board(P2, Color.PURPLE, ("education",))
    )


def test_the_demand_mandatorily_exchanges_the_whole_victim_hand_for_all_highest_cards() -> None:
    state = (
        _vulnerable()
        .hand(P1, ("tools", "construction", "canal-building"))
        .hand(P2, ("agriculture", "alchemy"))
        .build()
    )
    result = resolve_dogma(
        state,
        "machinery",
        choose_card("alchemy"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P2).hand == (CardId("canal-building"), CardId("construction"))
    assert set(result.state.player(P1).hand) == {CardId("agriculture"), CardId("tools")}
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)
    exchanges = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.EXCHANGE
    )
    assert len(exchanges) == 1
    assert len(exchanges[0].change.card_moves) == 4  # type: ignore[union-attr]


def test_an_empty_victim_hand_still_receives_the_activators_highest_cards() -> None:
    state = _vulnerable().hand(P1, ("tools",)).build()
    result = resolve_dogma(state, "machinery", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).hand == ()
    assert result.state.player(P2).hand == (CardId("tools"),)


def test_scoring_a_castle_precedes_the_independent_optional_red_splay() -> None:
    state = (
        _vulnerable()
        .board(P1, Color.RED, ("metalworking", "engineering"))
        .hand(P1, ("archery", "enterprise"))
        .hand(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(
        state,
        "machinery",
        choose_card("archery"),
        choose_branch("splay-left"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("archery"),)
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.LEFT
