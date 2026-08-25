"""DATABASES hidden score choices and rounded-up quantity snapshot."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_three_hidden_score_cards_require_two_victim_owned_returns() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("databases",))
        .score(P2, ("tools", "writing", "calendar"))
        .build()
    )
    result = resolve_dogma(
        state,
        "databases",
        choose_card("tools"),
        choose_card("calendar"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert [decision.chooser for decision in result.decisions] == [P2, P2]
    assert result.state.player(P2).score_pile == (CardId("writing"),)
    assert CardId("tools") in result.state.supply.pile(1)
    assert CardId("calendar") in result.state.supply.pile(2)
    returns = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "return"
    )
    assert len(returns) == 1
    assert len(returns[0].change.card_moves) == 2  # type: ignore[union-attr]
    assert returns[0].atomic_group_id is not None


def test_same_age_return_order_is_chosen_before_one_grouped_movement() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("databases",))
        .score(P2, ("tools", "writing", "sailing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "databases",
        choose_card("tools"),
        choose_card("writing"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert result.state.player(P2).score_pile == (CardId("sailing"),)
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2, P2)
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("writing")) < pile.index(CardId("tools"))
    returns = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "return"
    )
    assert len(returns) == 1
    assert len(returns[0].change.card_moves) == 2  # type: ignore[union-attr]


def test_an_empty_score_pile_rounds_to_zero_without_a_decision() -> None:
    state = scenario(REGISTRY).board(P1, Color.GREEN, ("databases",)).build()
    result = resolve_dogma(state, "databases", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert not result.decisions
