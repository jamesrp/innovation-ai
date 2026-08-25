"""DEMOCRACY: sequential scoped return-count records and per-dogma reset."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import (
    CardId,
    Color,
    PlayerId,
    SpecialAchievementId,
)

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _shared_state() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("democracy",))
        .board(P2, Color.PURPLE, ("philosophy",))
    )


def test_zero_returns_never_set_a_record() -> None:
    result = resolve_dogma(
        _shared_state().build(), "democracy", registry=REGISTRY, programs=PROGRAMS
    )
    assert result.decisions == ()
    assert not result.state.player(P1).score_pile
    assert not result.state.player(P2).score_pile


def test_equal_return_counts_reward_only_the_first_executor() -> None:
    state = (
        _shared_state()
        .hand(P2, ("archery", "calendar"))
        .hand(P1, ("agriculture", "alchemy"))
        .supply(8, ("flight", "mobility"))
        .build()
    )
    result = resolve_dogma(
        state,
        "democracy",
        choose_card("archery"),
        choose_card("calendar"),
        choose_card("agriculture"),
        choose_card("alchemy"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).score_pile == (CardId("flight"),)
    assert not result.state.player(P1).score_pile
    grouped_returns = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "return"
    )
    assert len(grouped_returns) == 2
    assert all(len(event.change.card_moves) == 2 for event in grouped_returns)  # type: ignore[union-attr]


def test_a_prior_executor_with_preexisting_monument_does_not_corrupt_the_tie_record() -> None:
    state = (
        _shared_state()
        .hand(P2, ("archery", "calendar"))
        .hand(P1, ("agriculture", "alchemy"))
        .achievements(P2, special=(SpecialAchievementId.MONUMENT,))
        .counters(P2, scored=6)
        .supply(8, ("flight", "mobility"))
        .build()
    )
    result = resolve_dogma(
        state,
        "democracy",
        choose_card("archery"),
        choose_card("calendar"),
        choose_card("agriculture"),
        choose_card("alchemy"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert result.state.player(P2).score_pile == (CardId("flight"),)
    assert not result.state.player(P1).score_pile
    assert result.state.supply.pile(8)[0] == CardId("mobility")


def test_a_later_executor_returning_more_cards_sets_a_new_record() -> None:
    state = (
        _shared_state()
        .hand(P2, ("archery", "calendar"))
        .hand(P1, ("agriculture", "alchemy", "construction"))
        .supply(8, ("flight", "mobility"))
        .build()
    )
    result = resolve_dogma(
        state,
        "democracy",
        choose_card("archery"),
        choose_card("calendar"),
        choose_card("agriculture"),
        choose_card("alchemy"),
        choose_card("construction"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).score_pile == (CardId("flight"),)
    assert result.state.player(P1).score_pile == (CardId("mobility"),)


def test_return_history_resets_for_the_next_dogma_action() -> None:
    state = (
        _shared_state()
        .hand(P2, ("archery",))
        .hand(P1, ("agriculture", "alchemy"))
        .supply(6, ("atomic-theory",))
        .supply(8, ("flight", "mobility", "quantum-theory"))
        .build()
    )
    first = resolve_dogma(
        state,
        "democracy",
        choose_card("archery"),
        choose_card("agriculture"),
        choose_card("alchemy"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert first.state.player(P1).hand == (CardId("atomic-theory"),)
    second = resolve_dogma(
        first.state,
        "democracy",
        choose_card("atomic-theory"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(second.state.player(P1).score_pile) == {
        CardId("mobility"),
        CardId("quantum-theory"),
    }
