"""TOOLS: exact optional cost, canonical subset/order, second branch, sharing, and resume."""

from __future__ import annotations

from support import ScenarioBuilder, choose_branch, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.BLUE, ("tools",)).board(P2, Color.RED, ("archery",))


def test_fewer_than_three_hand_cards_cannot_pay_effect_one() -> None:
    state = _solo().hand(P1, ("agriculture", "writing")).build()
    result = resolve_dogma(state, "tools", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.qualifying_changes == 0


def test_the_exact_three_card_cost_can_be_declined() -> None:
    state = _solo().hand(P1, ("agriculture", "clothing", "writing")).build()
    result = resolve_dogma(state, "tools", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert set(result.state.player(P1).hand) == {
        CardId("agriculture"),
        CardId("clothing"),
        CardId("writing"),
    }


def test_paying_three_returns_in_chosen_order_then_draws_and_melds_a_three() -> None:
    state = (
        _solo()
        .hand(P1, ("agriculture", "clothing", "sailing", "writing"))
        .supply(3, ("alchemy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "tools",
        choose_branch("return-three"),
        choose_card("agriculture"),
        choose_card("clothing"),
        choose_card("sailing"),
        choose_card("sailing"),
        choose_card("clothing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("writing"),)
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("alchemy")
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("sailing")) < pile.index(CardId("clothing"))
    assert pile.index(CardId("clothing")) < pile.index(CardId("agriculture"))


def test_effect_two_can_return_a_three_then_draw_exactly_three_ones() -> None:
    state = _solo().hand(P1, ("alchemy",)).supply(1, ("agriculture", "clothing", "writing")).build()
    result = resolve_dogma(
        state, "tools", choose_card("alchemy"), registry=REGISTRY, programs=PROGRAMS
    )
    assert set(result.state.player(P1).hand) == {
        CardId("agriculture"),
        CardId("clothing"),
        CardId("writing"),
    }
    assert CardId("alchemy") in result.state.supply.pile(3)


def test_effect_two_can_be_declined_after_effect_one_is_infeasible() -> None:
    state = _solo().hand(P1, ("alchemy",)).build()
    result = resolve_dogma(state, "tools", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("alchemy"),)


def test_a_sharing_opponent_that_changes_nothing_gives_no_bonus() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools",))
        .board(P2, Color.BLUE, ("writing",))
        .build()
    )
    result = resolve_dogma(state, "tools", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).hand
    assert not result.state.player(P2).hand
