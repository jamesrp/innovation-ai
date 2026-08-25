"""ELECTRICITY: factory filtering, bulk return order, and reward count."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_only_factoryless_tops_return_and_same_age_order_is_chosen_separately() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("electricity",))
        .board(P1, Color.BLUE, ("tools",))
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(8, ("flight", "rocketry"))
        .build()
    )
    result = resolve_dogma(
        state,
        "electricity",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("electricity")
    assert not result.state.player(P1).board.stack(Color.BLUE).cards
    assert not result.state.player(P1).board.stack(Color.RED).cards
    assert result.state.player(P1).hand == (CardId("flight"), CardId("rocketry"))
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("tools")) < pile.index(CardId("archery"))
    assert len(result.decisions) == 1
    assert result.decisions[0].chooser is P1


def test_no_factoryless_top_cards_means_no_returns_and_no_draws() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("electricity",))
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(state, "electricity", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).hand
