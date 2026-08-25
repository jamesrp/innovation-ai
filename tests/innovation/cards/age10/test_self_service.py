"""SELF SERVICE nested top-card execution and achievement-count win."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, NormalAchievementId, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_another_top_cards_non_demand_effects_execute_nested() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("self-service",))
        .board(P1, Color.YELLOW, ("globalization",))
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(
        state,
        "self-service",
        choose_card("globalization"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("canning"),)
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("code-of-laws")
    nested = tuple(
        event for event in result.events if event.source_card_id == CardId("globalization")
    )
    assert nested and all(
        event.nested and not event.demand and not event.shared for event in nested
    )


def test_more_achievements_than_the_opponent_wins() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("self-service",))
        .achievements(P1, normal=(NormalAchievementId.AGE_1,))
        .build()
    )
    result = resolve_dogma(state, "self-service", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P1,)
