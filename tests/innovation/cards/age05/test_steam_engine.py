"""STEAM ENGINE: ordered draw/tucks, bottom-yellow scoring, and sharing."""

from __future__ import annotations

from support import ScenarioBuilder, resolve_dogma, scenario

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


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("agriculture", "steam-engine"))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_two_draws_are_tucked_in_supply_order_then_the_new_bottom_yellow_scores() -> None:
    state = _solo().supply(4, ("anatomy", "colonialism")).build()
    result = resolve_dogma(state, "steam-engine", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == (CardId("anatomy"),)
    assert result.state.player(P1).board.stack(Color.YELLOW).cards == (
        CardId("agriculture"),
        CardId("steam-engine"),
    )
    assert result.state.player(P1).board.stack(Color.RED).bottom == CardId("colonialism")


def test_non_yellow_draws_leave_the_original_bottom_to_be_scored() -> None:
    state = _solo().supply(4, ("colonialism", "enterprise")).build()
    result = resolve_dogma(state, "steam-engine", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == (CardId("agriculture"),)
    assert result.state.player(P1).board.stack(Color.YELLOW).cards == (CardId("steam-engine"),)


def test_the_complete_two_pair_instruction_shares_one_atomic_group() -> None:
    state = _solo().supply(4, ("colonialism", "enterprise")).build()
    result = resolve_dogma(state, "steam-engine", registry=REGISTRY, programs=PROGRAMS)
    events = tuple(
        event
        for event in result.events
        if CardId("colonialism") in event.card_ids or CardId("enterprise") in event.card_ids
    )
    assert len(events) == 4
    assert len({event.atomic_group_id for event in events}) == 1


def test_sixth_achievement_waits_until_both_required_tucks_finish() -> None:
    state = (
        _solo()
        .achievements(P1, normal=tuple(NormalAchievementId)[:5])
        .counters(P1, tucked=5)
        .supply(4, ("colonialism", "enterprise"))
        .build()
    )
    result = resolve_dogma(state, "steam-engine", registry=REGISTRY, programs=PROGRAMS)

    assert result.status is EffectStatus.TERMINAL
    assert SpecialAchievementId.MONUMENT in result.state.player(P1).special_achievements
    assert result.state.player(P1).board.stack(Color.RED).bottom == CardId("colonialism")
    assert result.state.player(P1).board.stack(Color.PURPLE).bottom == CardId("enterprise")
    # The later bottom-yellow score is outside the completed tuck instruction and is interrupted.
    assert not result.state.player(P1).score_pile


def test_an_equal_factory_opponent_shares_first_and_causes_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("steam-engine",))
        .board(P2, Color.RED, ("colonialism",))
        .supply(
            4,
            ("anatomy", "enterprise", "navigation", "gunpowder", "invention"),
        )
        .build()
    )
    result = resolve_dogma(state, "steam-engine", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).score_pile == (CardId("anatomy"),)
    assert result.state.player(P1).score_pile == (CardId("steam-engine"),)
    assert result.state.player(P1).hand == (CardId("invention"),)
