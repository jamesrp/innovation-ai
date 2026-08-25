"""INDUSTRIALIZATION: factory-colour quantity snapshot and optional right splay."""

from __future__ import annotations

from support import decline, resolve_dogma, scenario

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


def test_factory_colour_count_is_snapshotted_before_new_tucks_expose_factories() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("industrialization",))
        .board(P1, Color.BLUE, ("chemistry",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(6, ("canning", "classification", "democracy"))
        .build()
    )
    result = resolve_dogma(
        state, "industrialization", decline(), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("canning")
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("classification")
    assert result.state.supply.pile(6)[0] == CardId("democracy")
    tuck_moves = tuple(
        move
        for event in result.events
        if event.change is not None and event.change.kind.value == "tuck"
        for move in event.change.card_moves
    )
    assert len(tuck_moves) == 2
    grouped = tuple(
        event
        for event in result.events
        if CardId("canning") in event.card_ids or CardId("classification") in event.card_ids
    )
    assert len(grouped) == 4
    assert len({event.atomic_group_id for event in grouped}) == 1


def test_sixth_achievement_waits_until_every_snapshotted_tuck_finishes() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("industrialization",))
        .board(P1, Color.BLUE, ("chemistry",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .achievements(P1, normal=tuple(NormalAchievementId)[:5])
        .counters(P1, tucked=5)
        .supply(6, ("canning", "classification"))
        .build()
    )
    result = resolve_dogma(state, "industrialization", registry=REGISTRY, programs=PROGRAMS)

    assert result.status is EffectStatus.TERMINAL
    assert SpecialAchievementId.MONUMENT in result.state.player(P1).special_achievements
    assert result.state.player(P1).board.stack(Color.YELLOW).bottom == CardId("canning")
    assert result.state.player(P1).board.stack(Color.GREEN).bottom == CardId("classification")
    assert result.decisions == ()
