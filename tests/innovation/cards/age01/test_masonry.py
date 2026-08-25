"""MASONRY: canonical castle subset, meld ordering, linked Monument, and sixth-achievement win."""

from __future__ import annotations

from support import choose_card, finish, resolve_dogma, scenario

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


def _solo():  # type: ignore[no-untyped-def]
    return (
        scenario(REGISTRY).board(P1, Color.YELLOW, ("masonry",)).board(P2, Color.BLUE, ("pottery",))
    )


def _four_castles():  # type: ignore[no-untyped-def]
    return ("archery", "city-states", "domestication", "tools")


def test_no_castle_cards_means_no_decision_and_no_claim() -> None:
    state = _solo().hand(P1, ("agriculture", "writing")).build()
    result = resolve_dogma(state, "masonry", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).special_achievements


def test_the_optional_subset_can_be_empty() -> None:
    state = _solo().hand(P1, ("archery", "tools")).build()
    result = resolve_dogma(state, "masonry", finish(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("archery"), CardId("tools"))
    assert not result.state.player(P1).special_achievements


def test_melding_four_castle_cards_claims_monument() -> None:
    state = _solo().hand(P1, _four_castles()).build()
    result = resolve_dogma(
        state,
        "masonry",
        *(choose_card(card) for card in _four_castles()),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert SpecialAchievementId.MONUMENT in result.state.player(P1).special_achievements
    assert not result.state.player(P1).hand
    assert any(event.achievement_id is SpecialAchievementId.MONUMENT for event in result.events)


def test_same_color_melds_have_a_separate_owner_order_choice() -> None:
    state = _solo().hand(P1, ("archery", "oars", "tools")).build()
    result = resolve_dogma(
        state,
        "masonry",
        choose_card("archery"),
        choose_card("oars"),
        finish(),
        choose_card("oars"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    order = result.decisions[-1]
    assert order.chooser is P1
    assert result.state.player(P1).board.stack(Color.RED).cards == (
        CardId("oars"),
        CardId("archery"),
    )


def test_a_linked_monument_as_the_sixth_achievement_stops_immediately() -> None:
    state = (
        _solo()
        .hand(P1, _four_castles())
        .achievements(P1, normal=tuple(NormalAchievementId)[:5])
        .build()
    )
    result = resolve_dogma(
        state,
        "masonry",
        *(choose_card(card) for card in _four_castles()),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None and result.terminal.winners == (P1,)
    assert SpecialAchievementId.MONUMENT in result.state.player(P1).special_achievements
