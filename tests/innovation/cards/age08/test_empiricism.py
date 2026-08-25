"""EMPIRICISM color-pair reveal handling and direct twenty-bulb victory."""

from __future__ import annotations

from support import choose_branch, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseBranchAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_a_matching_revealed_nine_is_melded_and_its_color_may_splay_up() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("empiricism",))
        .board(P1, Color.GREEN, ("compass", "paper"))
        .supply(9, ("collaboration",))
        .build()
    )
    result = resolve_dogma(
        state,
        "empiricism",
        choose_branch("blue-green"),
        choose_branch("splay"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    pair_actions = result.decisions[0].legal_actions
    assert len([action for action in pair_actions if isinstance(action, ChooseBranchAction)]) == 10
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("collaboration")
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.UP
    assert any(event.kind is EffectEventKind.REVEAL for event in result.events)
    assert not result.state.revealed


def test_twenty_or_more_visible_bulbs_award_an_immediate_direct_win() -> None:
    state = (
        scenario(REGISTRY)
        .board(
            P1,
            Color.PURPLE,
            ("philosophy", "education", "astronomy", "democracy", "empiricism"),
            splay=SplayDirection.UP,
        )
        .board(
            P1,
            Color.BLUE,
            ("tools", "writing", "experimentation", "atomic-theory", "evolution"),
            splay=SplayDirection.UP,
        )
        .supply(9, ("collaboration",))
        .build()
    )
    result = resolve_dogma(
        state,
        "empiricism",
        choose_branch("red-yellow"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P1,)
    assert CardId("collaboration") in result.state.player(P1).hand
