"""EXPERIMENTATION: atomic draw/meld, upward fallback, and shared execution."""

from __future__ import annotations

from support import ScenarioBuilder, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("experimentation",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_the_drawn_five_is_melded_atomically() -> None:
    state = _solo().supply(5, ("banking",)).build()
    result = resolve_dogma(state, "experimentation", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("banking")
    assert not result.state.player(P1).hand
    groups = {
        event.atomic_group_id for event in result.events if CardId("banking") in event.card_ids
    }
    assert len(groups) == 1


def test_an_empty_five_supply_uses_the_normal_upward_fallback() -> None:
    # Consume every age-5 card except Statistics, which becomes the hidden normal achievement.
    age_fives = tuple(
        card.id for card in REGISTRY.cards if card.age == 5 and card.id != CardId("statistics")
    )
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("experimentation",))
        .board(P2, Color.RED, ("metalworking",))
        .score(P1, age_fives)
        .supply(6, ("democracy",))
        .build()
    )
    result = resolve_dogma(state, "experimentation", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("democracy")


def test_a_stronger_bulb_opponent_melds_first_and_causes_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("experimentation",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .supply(5, ("banking", "coal", "chemistry"))
        .build()
    )
    result = resolve_dogma(state, "experimentation", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).board.stack(Color.GREEN).top == CardId("banking")
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("coal")
    assert result.state.player(P1).hand == (CardId("chemistry"),)
