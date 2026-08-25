"""REFRIGERATION: floor-half demand quantity and optional score."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _base() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("refrigeration",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_five_cards_returns_exactly_two_chosen_by_the_victim() -> None:
    state = (
        _base().hand(P2, ("tools", "writing", "canal-building", "construction", "alchemy")).build()
    )
    result = resolve_dogma(
        state,
        "refrigeration",
        choose_card("tools"),
        choose_card("construction"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2)
    assert set(result.state.player(P2).hand) == {
        CardId("writing"),
        CardId("canal-building"),
        CardId("alchemy"),
    }


def test_one_card_rounds_down_to_zero_returns() -> None:
    state = _base().hand(P2, ("tools",)).build()
    result = resolve_dogma(state, "refrigeration", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).hand == (CardId("tools"),)


def test_the_activator_may_score_one_of_their_remaining_hand_cards() -> None:
    state = _base().hand(P1, ("writing", "tools")).build()
    result = resolve_dogma(
        state,
        "refrigeration",
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("writing"),)
    assert result.state.player(P1).hand == (CardId("tools"),)
