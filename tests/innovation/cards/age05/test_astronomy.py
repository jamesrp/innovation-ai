"""ASTRONOMY: colour repeat, reveal cleanup, and the linked Universe route."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SpecialAchievementId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("astronomy",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_green_and_blue_draws_are_melded_before_the_effect_repeats() -> None:
    state = _solo().supply(6, ("classification", "atomic-theory", "democracy")).build()
    result = resolve_dogma(state, "astronomy", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("classification")
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("atomic-theory")
    assert result.state.player(P1).hand == (CardId("democracy"),)
    assert result.state.revealed == ()
    assert SpecialAchievementId.UNIVERSE in result.state.player(P1).special_achievements
    assert_conserved(result.state, REGISTRY)


def test_a_non_green_non_blue_draw_stops_without_melding() -> None:
    state = (
        _solo().board(P1, Color.RED, ("oars",)).supply(6, ("democracy", "classification")).build()
    )
    result = resolve_dogma(state, "astronomy", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("democracy"),)
    assert result.state.player(P1).board.stack(Color.GREEN).top is None
    assert SpecialAchievementId.UNIVERSE not in result.state.player(P1).special_achievements


def test_a_purple_only_board_satisfies_the_empty_universal_predicate() -> None:
    """Decision 10: no non-purple top cards means the Astronomy condition is true."""

    state = _solo().supply(6, ("democracy",)).build()
    result = resolve_dogma(state, "astronomy", registry=REGISTRY, programs=PROGRAMS)
    assert SpecialAchievementId.UNIVERSE in result.state.player(P1).special_achievements


def test_one_low_non_purple_top_card_blocks_the_linked_claim() -> None:
    state = _solo().board(P1, Color.GREEN, ("sailing",)).supply(6, ("democracy",)).build()
    result = resolve_dogma(state, "astronomy", registry=REGISTRY, programs=PROGRAMS)
    assert SpecialAchievementId.UNIVERSE not in result.state.player(P1).special_achievements


def test_an_equal_or_stronger_opponent_executes_each_non_demand_effect_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("astronomy",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .supply(6, ("democracy", "emancipation"))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(state, "astronomy", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).hand == (CardId("democracy"),)
    assert set(result.state.player(P1).hand) == {CardId("emancipation"), CardId("physics")}
    assert SpecialAchievementId.UNIVERSE in result.state.player(P2).special_achievements
