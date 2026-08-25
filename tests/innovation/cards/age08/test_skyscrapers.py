"""SKYSCRAPERS transfer destination and original-pile cleanup."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
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


def test_chosen_clock_top_moves_to_activator_then_beneath_scores_and_others_return() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("skyscrapers",))
        .board(P2, Color.GREEN, ("sailing", "compass", "paper", "mass-media"))
        .board(P2, Color.BLUE, ("tools", "quantum-theory"))
        .build()
    )
    result = resolve_dogma(
        state,
        "skyscrapers",
        choose_card("mass-media"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    offered = {
        action.card_id
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseCardAction)
    }
    assert offered == {CardId("mass-media"), CardId("quantum-theory")}
    compound = tuple(
        event
        for event in result.events
        if set(event.card_ids) & {CardId("paper"), CardId("sailing"), CardId("compass")}
    )
    assert len(compound) == 2
    assert compound[0].atomic_group_id is not None
    assert len({event.atomic_group_id for event in compound}) == 1
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("mass-media")
    assert result.state.player(P2).score_pile == (CardId("paper"),)
    assert not result.state.player(P2).board.stack(Color.GREEN).cards
    assert CardId("sailing") in result.state.supply.pile(1)
    assert CardId("compass") in result.state.supply.pile(3)
    assert result.state.player(P2).board.stack(Color.BLUE).top == CardId("quantum-theory")


def test_sixth_achievement_is_checked_only_after_all_required_returns() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("skyscrapers",))
        .board(P2, Color.GREEN, ("sailing", "compass", "paper", "mass-media"))
        .achievements(P2, normal=tuple(NormalAchievementId)[:5])
        .counters(P2, scored=5)
        .build()
    )
    result = resolve_dogma(
        state,
        "skyscrapers",
        choose_card("mass-media"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None and result.terminal.winners == (P2,)
    assert SpecialAchievementId.MONUMENT in result.state.player(P2).special_achievements
    assert not result.state.player(P2).board.stack(Color.GREEN).cards
    assert CardId("sailing") in result.state.supply.pile(1)
    assert CardId("compass") in result.state.supply.pile(3)
