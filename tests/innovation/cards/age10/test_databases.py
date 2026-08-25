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


def test_an_empty_score_pile_rounds_to_zero_without_a_decision() -> None:
    state = scenario(REGISTRY).board(P1, Color.GREEN, ("databases",)).build()
    result = resolve_dogma(state, "databases", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert not result.decisions
