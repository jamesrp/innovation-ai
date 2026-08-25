"""EVOLUTION: both optional branches and the empty-score value-zero rule."""

from __future__ import annotations

from support import ScenarioBuilder, choose_branch, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _base() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("evolution",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_score_eight_branch_scores_before_offering_the_return() -> None:
    state = _base().score(P1, ("tools",)).supply(8, ("flight",)).build()
    result = resolve_dogma(
        state,
        "evolution",
        choose_branch("score-eight"),
        choose_card("flight"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id for action in result.decisions[1].legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("tools"), CardId("flight")}
    assert result.state.player(P1).score_pile == (CardId("tools"),)


def test_draw_above_an_empty_score_pile_draws_a_one() -> None:
    state = _base().supply(1, ("writing",)).build()
    result = resolve_dogma(
        state,
        "evolution",
        choose_branch("draw-above-score"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("writing"),)


def test_the_entire_alternative_can_be_declined() -> None:
    state = _base().build()
    result = resolve_dogma(
        state,
        "evolution",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert not result.state.player(P1).hand
    assert not result.state.player(P1).score_pile
    assert result.qualifying_changes == 0
