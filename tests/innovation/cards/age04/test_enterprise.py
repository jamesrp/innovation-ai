"""ENTERPRISE: filtered top-card demand, victim reward, board transfer, and splay sharing."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P2, Color.YELLOW, ("medicine",))
    )


def test_the_transferred_top_joins_the_activators_board_and_the_victim_melds_a_four() -> None:
    state = _vulnerable().supply(4, ("gunpowder",)).build()
    result = resolve_dogma(
        state,
        "enterprise",
        choose_card("medicine"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("medicine")
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("gunpowder")
    assert not result.state.player(P2).hand


def test_only_non_purple_crown_tops_are_offered_to_the_victim() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P1, Color.GREEN, ("navigation",))
        .board(P2, Color.YELLOW, ("medicine",))
        .board(P2, Color.GREEN, ("paper",))
        .board(P2, Color.PURPLE, ("city-states",))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "enterprise",
        choose_card("paper"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseCardAction)
    }
    assert offered == {CardId("medicine"), CardId("paper")}
    assert result.decisions[0].chooser is P2
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("paper")


def test_a_singleton_green_stack_is_still_a_legal_splay_choice() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P1, Color.GREEN, ("sailing",))
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    result = resolve_dogma(
        state,
        "enterprise",
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.NONE


def test_an_equal_crown_opponent_is_immune_but_shares_the_second_effect_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P1, Color.GREEN, ("invention",))
        .board(P2, Color.GREEN, ("navigation",))
        .build()
    )
    result = resolve_dogma(
        state,
        "enterprise",
        choose_color(Color.GREEN),
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("navigation")
    assert not result.state.player(P2).hand
