"""MONOTHEISM: relational demand, conditional victim tuck, immunity, sharing, and partiality."""

from __future__ import annotations

from support import DogmaResult, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _board_card_count(result: DogmaResult, player_id: PlayerId) -> int:
    return sum(len(stack.cards) for stack in result.state.player(player_id).board.stacks)


def test_a_different_color_top_transfers_then_the_victim_draws_and_tucks() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("monotheism",))
        .board(P2, Color.BLUE, ("calendar",))
        .supply(1, ("agriculture", "archery"))
        .build()
    )
    result = resolve_dogma(
        state,
        "monotheism",
        choose_card("calendar"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("calendar"),)
    assert _board_card_count(result, P2) == 1
    assert _board_card_count(result, P1) == 2


def test_a_color_anywhere_on_the_activator_board_excludes_that_victim_top() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("monotheism",))
        .board(P1, Color.GREEN, ("the-wheel", "sailing"))
        .board(P2, Color.GREEN, ("currency",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "monotheism", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("currency")
    assert not result.state.player(P1).score_pile
    assert _board_card_count(result, P1) == 4


def test_equal_castles_skip_the_demand_but_share_the_second_effect() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("monotheism",))
        .board(P2, Color.RED, ("construction",))
        .supply(1, ("agriculture", "archery"))
        .supply(2, ("calendar",))
        .build()
    )
    result = resolve_dogma(state, "monotheism", registry=REGISTRY, programs=PROGRAMS)
    assert _board_card_count(result, P2) == 2
    assert _board_card_count(result, P1) == 2
    assert result.state.player(P1).hand == (CardId("calendar"),)


def test_missing_victim_top_still_allows_the_independent_second_effect() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("monotheism",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "monotheism", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == ()
    assert _board_card_count(result, P1) == 2
