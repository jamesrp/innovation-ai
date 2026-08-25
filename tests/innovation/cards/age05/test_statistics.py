"""STATISTICS: all tied highest scores, empty partial execution, and sharing."""

from __future__ import annotations

from support import ScenarioBuilder, choose_color, decline, resolve_dogma, scenario

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
        .board(P1, Color.YELLOW, ("statistics",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_all_tied_highest_score_cards_transfer_to_the_victims_hand() -> None:
    state = _vulnerable().score(P2, ("tools", "calendar", "construction")).build()
    result = resolve_dogma(
        state,
        "statistics",
        choose_color(Color.YELLOW),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.decisions[0].chooser is P1  # the demand itself needs no tie choice
    assert set(result.state.player(P2).hand) == {CardId("calendar"), CardId("construction")}
    assert result.state.player(P2).score_pile == (CardId("tools"),)


def test_an_empty_score_pile_makes_the_demand_a_no_op() -> None:
    state = _vulnerable().build()
    result = resolve_dogma(state, "statistics", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert len(result.decisions) == 1
    assert not result.state.player(P2).hand
    assert result.qualifying_changes == 0


def test_a_singleton_yellow_stack_is_still_a_legal_splay_choice() -> None:
    state = _vulnerable().build()
    result = resolve_dogma(
        state,
        "statistics",
        choose_color(Color.YELLOW),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.YELLOW).splay is SplayDirection.NONE


def test_an_immune_opponent_shares_the_splay_first_and_earns_a_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("statistics",))
        .board(P2, Color.YELLOW, ("agriculture", "anatomy"))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "statistics",
        choose_color(Color.YELLOW),
        choose_color(Color.YELLOW),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P2).board.stack(Color.YELLOW).splay is SplayDirection.RIGHT
    assert result.state.player(P1).hand == (CardId("physics"),)
