"""BIOENGINEERING cross-player transfer and unique visible-leaf victory."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_a_leafy_opponent_top_card_transfers_to_the_executors_score() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("bioengineering",))
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "bioengineering",
        choose_card("pottery"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("pottery"),)
    assert not result.state.player(P2).board.stack(Color.BLUE).cards


def test_if_anyone_has_fewer_than_three_leaves_the_unique_leaf_leader_wins() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("bioengineering",))
        .board(P1, Color.YELLOW, ("agriculture",))
        .board(P2, Color.RED, ("archery",))
        .build()
    )
    result = resolve_dogma(state, "bioengineering", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P1,)
