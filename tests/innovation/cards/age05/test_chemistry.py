"""CHEMISTRY: optional splay, computed draw value, hidden score return, and termination."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("chemistry",))
        .board(P2, Color.YELLOW, ("agriculture",))
    )


def test_the_newly_scored_six_can_be_returned_immediately() -> None:
    state = _solo().supply(6, ("classification",)).build()
    result = resolve_dogma(
        state,
        "chemistry",
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.NONE
    assert not result.state.player(P1).score_pile
    assert CardId("classification") in result.state.supply.pile(6)


def test_the_executor_may_return_an_old_score_card_and_keep_the_reward() -> None:
    state = _solo().score(P1, ("tools",)).supply(6, ("classification",)).build()
    result = resolve_dogma(
        state,
        "chemistry",
        decline(),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    return_decision = result.decisions[-1]
    assert return_decision.chooser is P1
    offered = {
        action.card_id for action in return_decision.legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("tools"), CardId("classification")}
    assert result.state.player(P1).score_pile == (CardId("classification"),)


def test_a_two_card_blue_stack_can_be_splayed_right() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools", "chemistry"))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(6, ("classification",))
        .build()
    )
    result = resolve_dogma(
        state,
        "chemistry",
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.RIGHT


def test_highest_top_value_ten_requests_eleven_and_ends_immediately() -> None:
    state = _solo().board(P1, Color.PURPLE, ("a-i",)).build()
    result = resolve_dogma(state, "chemistry", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert len(result.decisions) == 1
    assert not result.state.pending_effects


def test_an_equal_factory_opponent_shares_effect_two_before_the_activator() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("chemistry",))
        .board(P2, Color.RED, ("colonialism",))
        .supply(5, ("physics", "banking"))
        .supply(6, ("classification",))
        .build()
    )
    result = resolve_dogma(
        state,
        "chemistry",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P1,)
    assert not result.state.player(P2).score_pile
    assert not result.state.player(P1).score_pile
    assert result.state.player(P1).hand == (CardId("banking"),)
