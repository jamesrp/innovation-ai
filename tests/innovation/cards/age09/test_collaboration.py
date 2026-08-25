"""COLLABORATION cross-player reveal choice and immediate victory."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_activator_chooses_between_the_victims_two_revealed_draws() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("collaboration",))
        .hand(P2, ("tools",))
        .supply(9, ("composites", "ecology"))
        .build()
    )
    result = resolve_dogma(
        state,
        "collaboration",
        choose_card("composites"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    decision = result.decisions[0]
    assert decision.chooser is P1
    assert decision.executor is P2
    assert {
        action.card_id for action in decision.legal_actions if isinstance(action, ChooseCardAction)
    } == {CardId("composites"), CardId("ecology")}
    assert set(decision.observation.revealed_cards) == {
        CardId("composites"),
        CardId("ecology"),
    }
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("composites")
    assert result.state.player(P2).board.stack(Color.YELLOW).top == CardId("ecology")
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert not result.state.revealed


def test_ten_green_board_cards_award_the_executor_an_immediate_win() -> None:
    greens = (
        "sailing",
        "the-wheel",
        "clothing",
        "currency",
        "mapmaking",
        "compass",
        "paper",
        "invention",
        "banking",
        "collaboration",
    )
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, greens)
        .board(P2, Color.GREEN, ("self-service",))
        .build()
    )
    result = resolve_dogma(
        state,
        "collaboration",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P1,)
