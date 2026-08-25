"""BANKING: demand targeting/pronouns and legal fixed-colour splay choices."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY).board(P1, Color.GREEN, ("banking",)).board(P2, Color.BLUE, ("pottery",))
    )


def test_the_victim_transfers_a_qualifying_top_then_draws_and_scores() -> None:
    state = (
        _vulnerable()
        .board(P2, Color.RED, ("colonialism",))
        .board(P2, Color.YELLOW, ("canning",))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "banking",
        choose_card("canning"),
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.decisions[0].chooser is P2
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("canning")
    assert result.state.player(P2).score_pile == (CardId("physics"),)
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.NONE


def test_green_and_non_factory_tops_are_not_demand_targets() -> None:
    state = (
        _vulnerable()
        .board(P2, Color.GREEN, ("electricity",))
        .board(P2, Color.RED, ("colonialism",))
        .board(P2, Color.YELLOW, ("anatomy",))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "banking",
        choose_card("colonialism"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("colonialism")}


def test_no_qualifying_top_skips_the_demand_reward() -> None:
    state = _vulnerable().supply(5, ("physics",)).build()
    result = resolve_dogma(state, "banking", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert len(result.decisions) == 1  # only Banking's own optional green splay
    assert not result.state.player(P2).score_pile
    assert CardId("physics") in result.state.supply.pile(5)


def test_an_equal_opponent_ignores_the_demand_and_shares_the_splay_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("banking",))
        .board(P2, Color.GREEN, ("mapmaking", "currency"))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "banking",
        choose_color(Color.GREEN),
        choose_color(Color.GREEN),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P2).board.stack(Color.GREEN).splay is SplayDirection.RIGHT
    # The shared effective splay earns exactly one free Draw for the activator.
    assert result.state.player(P1).hand == (CardId("physics"),)
