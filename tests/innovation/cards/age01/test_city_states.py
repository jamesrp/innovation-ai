"""CITY STATES: threshold guard, public top-card choice, transfer pronouns, and immunity."""

from __future__ import annotations

from support import assert_conserved, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_fewer_than_four_castles_does_nothing() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("city-states",))
        .board(P2, Color.RED, ("archery",))
        .build()
    )
    result = resolve_dogma(state, "city-states", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.qualifying_changes == 0


def test_the_victim_chooses_a_castle_top_to_transfer_then_draws() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("city-states",))
        .board(P2, Color.RED, ("metalworking",))
        .board(P2, Color.GREEN, ("the-wheel",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        "city-states",
        choose_card("the-wheel"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    decision = result.decisions[0]
    assert decision.chooser is P2 and decision.executor is P2
    assert {action.card_id for action in decision.legal_actions if hasattr(action, "card_id")} == {
        CardId("metalworking"),
        CardId("the-wheel"),
    }
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("the-wheel")
    assert result.state.player(P2).hand == (CardId("agriculture"),)
    assert_conserved(result.state, REGISTRY)


def test_a_successful_transfer_and_draw_have_demand_provenance() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("city-states",))
        .board(P2, Color.RED, ("metalworking",))
        .board(P2, Color.GREEN, ("the-wheel",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        "city-states",
        choose_card("metalworking"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert all(event.demand and event.executor is P2 for event in result.events if event.changed)
    assert not result.state.player(P1).hand, "demands never award a sharing bonus"


def test_equal_crowns_make_the_opponent_immune() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("city-states",))
        .board(P2, Color.GREEN, ("sailing",))
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    result = resolve_dogma(state, "city-states", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("metalworking")
    assert not result.state.player(P1).board.stack(Color.RED).cards
