"""SERVICES demand transfers between score, hand, and board zones."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("services",))
        .board(P1, Color.RED, ("composites",))
        .board(P1, Color.BLUE, ("genetics",))
    )


def test_all_highest_scores_move_to_activator_hand_then_victim_takes_a_leafless_top() -> None:
    state = _vulnerable().score(P2, ("calendar", "alchemy", "compass")).build()
    result = resolve_dogma(
        state,
        "services",
        choose_card("composites"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert {CardId("alchemy"), CardId("compass")} <= set(result.state.player(P1).hand)
    assert CardId("composites") in result.state.player(P2).hand
    assert not result.state.player(P1).board.stack(Color.RED).cards
    assert result.state.player(P2).score_pile == (CardId("calendar"),)


def test_an_empty_score_pile_skips_the_conditional_board_transfer() -> None:
    result = resolve_dogma(
        _vulnerable().build(),
        "services",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert not result.decisions
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("composites")
