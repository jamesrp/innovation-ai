"""VACCINATION: all-lowest demand return and demand-history follow-up."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("vaccination",))
        .board(P2, Color.BLUE, ("tools",))
    )


def test_all_tied_lowest_cards_return_then_both_conditional_melds_occur() -> None:
    state = (
        _vulnerable()
        .score(P2, ("writing", "sailing", "alchemy"))
        .supply(6, ("classification",))
        .supply(7, ("bicycle",))
        .build()
    )
    result = resolve_dogma(
        state,
        "vaccination",
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).score_pile == (CardId("alchemy"),)
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("classification")
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("bicycle")
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("writing")) < pile.index(CardId("sailing"))
    assert tuple(decision.chooser for decision in result.decisions) == (P2,)


def test_no_demand_return_means_neither_conditional_draw_occurs() -> None:
    state = _vulnerable().supply(6, ("classification",)).supply(7, ("bicycle",)).build()
    result = resolve_dogma(state, "vaccination", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.GREEN).top is None
    assert result.state.player(P1).board.stack(Color.GREEN).top is None
    assert result.state.supply.pile(6)[0] == CardId("classification")
    assert result.state.supply.pile(7)[0] == CardId("bicycle")


def test_demand_immunity_does_not_create_false_history_for_the_follow_up() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("vaccination",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .score(P2, ("writing",))
        .supply(7, ("bicycle",))
        .build()
    )
    result = resolve_dogma(state, "vaccination", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).score_pile == (CardId("writing"),)
    assert result.state.supply.pile(7)[0] == CardId("bicycle")
