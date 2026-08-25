"""COAL: atomic draw/tuck, optional splay, and top-plus-beneath scoring."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("oars", "coal"))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_scoring_a_top_card_also_scores_the_snapshotted_card_beneath_it() -> None:
    state = _solo().supply(5, ("physics",)).build()
    result = resolve_dogma(
        state,
        "coal",
        choose_color(Color.RED),
        choose_card("coal"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).score_pile) == {CardId("coal"), CardId("oars")}
    assert not result.state.player(P1).board.stack(Color.RED).cards
    assert result.state.player(P1).board.stack(Color.BLUE).bottom == CardId("physics")


def test_a_singleton_top_scores_even_though_it_has_no_card_beneath() -> None:
    state = _solo().board(P1, Color.YELLOW, ("agriculture",)).supply(5, ("physics",)).build()
    result = resolve_dogma(
        state,
        "coal",
        decline(),
        choose_card("agriculture"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("agriculture"),)
    assert result.state.player(P1).board.stack(Color.RED).cards == (
        CardId("oars"),
        CardId("coal"),
    )


def test_both_optional_effects_can_be_declined() -> None:
    state = _solo().supply(5, ("physics",)).build()
    result = resolve_dogma(
        state, "coal", decline(), decline(), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.NONE
    assert not result.state.player(P1).score_pile
    assert result.state.player(P1).board.stack(Color.BLUE).bottom == CardId("physics")


def test_the_draw_and_tuck_share_one_atomic_group() -> None:
    state = _solo().supply(5, ("physics",)).build()
    result = resolve_dogma(
        state, "coal", decline(), decline(), registry=REGISTRY, programs=PROGRAMS
    )
    physics_events = tuple(event for event in result.events if CardId("physics") in event.card_ids)
    assert len(physics_events) == 2
    assert len({event.atomic_group_id for event in physics_events}) == 1


def test_an_equal_factory_opponent_shares_all_three_effects_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("coal",))
        .board(P2, Color.RED, ("machine-tools",))
        .supply(5, ("physics", "banking", "statistics"))
        .build()
    )
    result = resolve_dogma(
        state,
        "coal",
        decline(),
        decline(),
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1, P2, P1)
    assert result.state.player(P2).board.stack(Color.BLUE).bottom == CardId("physics")
    assert result.state.player(P1).board.stack(Color.GREEN).bottom == CardId("banking")
    assert result.state.player(P1).hand == (CardId("statistics"),)
