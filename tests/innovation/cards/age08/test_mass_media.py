"""MASS MEDIA global score-pile return and unrestricted value choice."""

from __future__ import annotations

from support import choose_card, choose_color, choose_value, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseValueAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_chosen_value_returns_matching_cards_from_all_score_piles() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("mass-media",))
        .hand(P1, ("tools",))
        .score(P1, ("canal-building", "alchemy"))
        .score(P2, ("construction", "currency", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "mass-media",
        choose_card("tools"),
        choose_value(2),
        choose_card("currency"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    value_actions = result.decisions[1].legal_actions
    assert {
        action.value for action in value_actions if isinstance(action, ChooseValueAction)
    } == set(range(1, 11))
    assert result.decisions[2].chooser is P2
    assert result.state.supply.pile(2).index(CardId("currency")) < result.state.supply.pile(
        2
    ).index(CardId("construction"))
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)
    assert result.state.player(P2).score_pile == (CardId("writing"),)
    assert {CardId("canal-building"), CardId("construction"), CardId("currency")} <= set(
        result.state.supply.pile(2)
    )


def test_purple_stack_may_splay_up() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("mass-media",))
        .board(P1, Color.PURPLE, ("mysticism", "services"))
        .build()
    )
    result = resolve_dogma(
        state,
        "mass-media",
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.PURPLE).splay is SplayDirection.UP
