"""INDUSTRIALIZATION: factory-colour quantity snapshot and optional right splay."""

from __future__ import annotations

from support import decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

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
