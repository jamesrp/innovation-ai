"""ENCYCLOPEDIA: optional all-or-none highest-score meld."""

from __future__ import annotations

from support import choose_branch, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseBranchAction, ChooseCardAction
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
        .board(P1, Color.BLUE, ("encyclopedia",))
        .board(P2, Color.RED, ("metalworking",))
        .score(P1, ("tools", "banking", "measurement"))
        .build()
    )


def test_accepting_melds_every_highest_card_in_chosen_same_colour_order() -> None:
    result = resolve_dogma(
        _state(),
        "encyclopedia",
        choose_branch("meld-all-highest"),
        choose_card("banking"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    first = result.decisions[0]
    assert any(isinstance(action, ChooseBranchAction) for action in first.legal_actions)
    assert not any(isinstance(action, ChooseCardAction) for action in first.legal_actions)
    assert result.state.player(P1).score_pile == (CardId("tools"),)
    assert result.state.player(P1).board.stack(Color.GREEN).cards == (
        CardId("banking"),
        CardId("measurement"),
    )
    meld_events = tuple(event for event in result.events if event.change is not None)
    assert len(meld_events) == 1
    assert len(meld_events[0].change.card_moves) == 2  # type: ignore[union-attr]


def test_declining_leaves_the_complete_highest_set_in_the_score_pile() -> None:
    state = _state()
    result = resolve_dogma(state, "encyclopedia", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == state.player(P1).score_pile
    assert result.state.player(P1).board == state.player(P1).board
