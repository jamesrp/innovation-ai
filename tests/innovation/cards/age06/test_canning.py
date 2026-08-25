"""CANNING: optional atomic tuck, live factoryless-top scoring, and yellow splay."""

from __future__ import annotations

from support import choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _state() -> GameState:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("canning",))
        .board(P1, Color.BLUE, ("tools",))
        .board(P1, Color.GREEN, ("sailing",))
        .board(P1, Color.PURPLE, ("democracy",))
        .board(P1, Color.RED, ("machine-tools",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(6, ("classification",))
        .build()
    )


def test_accepting_tucks_then_scores_every_live_factoryless_top() -> None:
    result = resolve_dogma(
        _state(),
        "canning",
        choose_branch("draw-and-tuck"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).score_pile) == {
        CardId("tools"),
        CardId("sailing"),
        CardId("democracy"),
    }
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("classification")
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("canning")
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("machine-tools")


def test_declining_both_optional_instructions_changes_nothing() -> None:
    state = _state()
    result = resolve_dogma(
        state,
        "canning",
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board == state.player(P1).board
    assert not result.state.player(P1).score_pile
