"""CLOTHING: relational meld selection, live unique colors, atomic scoring, and sharing."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_no_different_hand_color_and_no_unique_board_color_is_a_no_op() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("clothing",))
        .board(P2, Color.GREEN, ("the-wheel",))
        .hand(P1, ("sailing",))
        .build()
    )
    result = resolve_dogma(state, "clothing", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.qualifying_changes == 0
    assert result.state.player(P1).hand == (CardId("sailing"),)


def test_a_newly_melded_color_counts_for_effect_two() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("clothing",))
        .board(P2, Color.RED, ("archery",))
        .hand(P1, ("writing",))
        .supply(1, ("agriculture", "masonry"))
        .build()
    )
    result = resolve_dogma(
        state, "clothing", choose_card("writing"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("writing")
    assert len(result.state.player(P1).score_pile) == 2


def test_only_colors_absent_from_the_opponents_board_are_counted() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("clothing",))
        .board(P1, Color.BLUE, ("tools",))
        .board(P2, Color.BLUE, ("writing",))
        .board(P2, Color.RED, ("archery",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "clothing", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == (CardId("agriculture"),)


def test_a_sharing_opponent_scores_first_and_causes_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("clothing",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(1, ("agriculture", "domestication", "masonry"))
        .build()
    )
    result = resolve_dogma(state, "clothing", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P2).score_pile) == 1
    assert len(result.state.player(P1).score_pile) == 1
    assert len(result.state.player(P1).hand) == 1
