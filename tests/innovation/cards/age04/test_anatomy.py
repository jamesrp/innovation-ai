"""ANATOMY: clarified score choice, matching top return, immunity, and hidden ownership."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.YELLOW, ("anatomy",)).board(P2, Color.RED, ("archery",))
    )


def test_clarification_allows_a_score_choice_that_avoids_the_board_return() -> None:
    state = (
        _vulnerable().board(P2, Color.GREEN, ("sailing",)).score(P2, ("tools", "calendar")).build()
    )
    result = resolve_dogma(
        state,
        "anatomy",
        choose_card("calendar"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    decision = result.decisions[0]
    assert decision.chooser is P2
    assert {
        action.card_id for action in decision.legal_actions if isinstance(action, ChooseCardAction)
    } == {CardId("tools"), CardId("calendar")}
    assert result.state.player(P2).score_pile == (CardId("tools"),)
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("sailing")


def test_the_victim_breaks_a_tie_between_matching_value_top_cards() -> None:
    state = _vulnerable().board(P2, Color.PURPLE, ("city-states",)).score(P2, ("tools",)).build()
    result = resolve_dogma(
        state,
        "anatomy",
        choose_card("city-states"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1  # the sole score card is automatic
    assert result.decisions[0].chooser is P2
    assert not result.state.player(P2).board.stack(Color.PURPLE).cards
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("archery")


def test_an_empty_score_pile_makes_the_demand_a_no_op() -> None:
    state = _vulnerable().build()
    result = resolve_dogma(state, "anatomy", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.qualifying_changes == 0


def test_equal_leaf_counts_make_the_opponent_immune() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("anatomy",))
        .board(P2, Color.BLUE, ("pottery",))
        .score(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "anatomy", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).score_pile == (CardId("tools"),)
