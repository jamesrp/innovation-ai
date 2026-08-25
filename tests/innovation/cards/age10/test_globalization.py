"""GLOBALIZATION leafy demand plus conditional unique-points victory."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_the_victim_returns_one_leafy_top_card() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("globalization",))
        .board(P2, Color.BLUE, ("pottery",))
        .board(P2, Color.GREEN, ("sailing",))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(
        state,
        "globalization",
        choose_card("pottery"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P2).board.stack(Color.BLUE).cards
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("sailing")
    assert CardId("canning") in result.state.player(P1).score_pile


def test_when_the_icon_guard_holds_the_unique_points_leader_can_be_the_opponent() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("globalization",))
        .board(P2, Color.RED, ("archery",))
        .score(P2, ("bicycle",))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(state, "globalization", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P2,)


def test_a_points_tie_ignores_the_win_effect() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("globalization",))
        .score(P2, ("atomic-theory",))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(state, "globalization", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.terminal is None
