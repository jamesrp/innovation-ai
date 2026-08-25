"""CONSTRUCTION: mandatory partial demand, immunity, linked Empire, and terminal claim."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
)

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("construction",))
        .board(P2, Color.BLUE, ("calendar",))
    )


def _empire_board() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("construction",))
        .board(P1, Color.BLUE, ("calendar",))
        .board(P1, Color.GREEN, ("currency",))
        .board(P1, Color.YELLOW, ("fermenting",))
        .board(P1, Color.PURPLE, ("philosophy",))
        .board(P2, Color.RED, ("archery",))
    )


def test_one_available_hand_card_is_transferred_before_the_independent_draw() -> None:
    state = _vulnerable().hand(P2, ("tools",)).supply(2, ("canal-building",)).build()
    result = resolve_dogma(
        state,
        "construction",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert result.state.player(P2).hand == (CardId("canal-building"),)
    assert tuple(decision.chooser for decision in result.decisions) == (P2,)


def test_the_victim_selects_exactly_two_when_more_are_available() -> None:
    state = (
        _vulnerable()
        .hand(P2, ("agriculture", "tools", "writing"))
        .supply(2, ("mapmaking",))
        .build()
    )
    result = resolve_dogma(
        state,
        "construction",
        choose_card("agriculture"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {CardId("agriculture"), CardId("tools")}
    assert set(result.state.player(P2).hand) == {CardId("writing"), CardId("mapmaking")}


def test_equal_castles_make_the_opponent_immune_to_the_demand() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("construction",))
        .board(P2, Color.PURPLE, ("monotheism",))
        .hand(P2, ("tools", "writing"))
        .build()
    )
    result = resolve_dogma(state, "construction", registry=REGISTRY, programs=PROGRAMS)
    assert set(result.state.player(P2).hand) == {CardId("tools"), CardId("writing")}
    assert not result.state.player(P1).hand


def test_the_only_five_top_card_player_claims_linked_empire() -> None:
    state = _empire_board().supply(2, ("canal-building",)).build()
    result = resolve_dogma(state, "construction", registry=REGISTRY, programs=PROGRAMS)
    assert SpecialAchievementId.EMPIRE in result.state.player(P1).special_achievements
    assert any(event.achievement_id is SpecialAchievementId.EMPIRE for event in result.events)


def test_a_shared_linked_claim_alone_earns_the_activator_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("construction",))
        .board(P2, Color.RED, ("archery",))
        .board(P2, Color.BLUE, ("calendar",))
        .board(P2, Color.GREEN, ("currency",))
        .board(P2, Color.YELLOW, ("fermenting",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .supply(2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(state, "construction", registry=REGISTRY, programs=PROGRAMS)
    assert SpecialAchievementId.EMPIRE in result.state.player(P2).special_achievements
    assert SpecialAchievementId.EMPIRE not in result.state.player(P1).special_achievements
    assert result.state.player(P1).hand == (CardId("canal-building"),)


def test_linked_empire_as_the_sixth_achievement_ends_immediately() -> None:
    state = (
        _empire_board()
        .achievements(P1, normal=tuple(NormalAchievementId)[:5])
        .supply(2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(state, "construction", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None and result.terminal.winners == (P1,)
    assert SpecialAchievementId.EMPIRE in result.state.player(P1).special_achievements
