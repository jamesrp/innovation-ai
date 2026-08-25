"""ROAD BUILDING: one-or-two clarification, meld order, linked transfers, partiality, replay."""

from __future__ import annotations

from support import ScenarioBuilder, choose_branch, choose_card, finish, resolve_dogma, scenario

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
        .board(P1, Color.RED, ("road-building",))
        .board(P2, Color.BLUE, ("calendar",))
    )


def test_an_empty_hand_partially_executes_without_a_decision() -> None:
    result = resolve_dogma(_solo().build(), "road-building", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("road-building")


def test_one_available_card_is_melded_without_offering_the_transfer_branch() -> None:
    state = _solo().hand(P1, ("tools",)).build()
    result = resolve_dogma(
        state,
        "road-building",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("tools")
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("road-building")


def test_the_official_clarification_allows_stopping_after_one_of_many_cards() -> None:
    state = _solo().hand(P1, ("agriculture", "tools", "writing")).build()
    result = resolve_dogma(
        state,
        "road-building",
        choose_card("agriculture"),
        finish(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("agriculture")
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}


def test_two_same_color_melds_use_a_separate_order_before_linked_transfers() -> None:
    state = _solo().board(P2, Color.GREEN, ("sailing",)).hand(P1, ("archery", "oars")).build()
    result = resolve_dogma(
        state,
        "road-building",
        choose_card("archery"),
        choose_card("oars"),
        choose_card("oars"),
        choose_branch("transfer-top-red"),
        registry=REGISTRY,
        programs=PROGRAMS,
        verify_resume=True,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P1, P1, P1, P1)
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("oars")
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("archery")
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("sailing")
    assert not result.state.player(P2).board.stack(Color.GREEN).cards


def test_missing_reciprocal_green_card_does_not_undo_the_red_transfer() -> None:
    state = _solo().hand(P1, ("archery", "tools")).build()
    result = resolve_dogma(
        state,
        "road-building",
        choose_card("archery"),
        choose_card("tools"),
        choose_branch("transfer-top-red"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("archery")
    assert not result.state.player(P1).board.stack(Color.GREEN).cards
