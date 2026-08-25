"""WRITING: deterministic value-two draw, upward fallback, sharing, and terminal exhaustion."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.BLUE, ("writing",)).board(P2, Color.RED, ("archery",))


def test_writing_draws_one_value_two_card_without_a_decision() -> None:
    state = _solo().supply(2, ("canal-building",)).build()
    result = resolve_dogma(state, "writing", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("canal-building"),)
    assert_conserved(result.state, REGISTRY)


def test_an_empty_two_pile_searches_upward() -> None:
    age_two = tuple(
        sorted(
            card.id for card in REGISTRY.cards if card.age == 2 and card.id != CardId("calendar")
        )
    )
    state = _solo().score(P2, age_two).supply(3, ("alchemy",)).build()
    assert not state.supply.pile(2)
    result = resolve_dogma(state, "writing", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("alchemy"),)


def test_equal_lightbulbs_share_and_award_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("writing",))
        .board(P2, Color.BLUE, ("tools",))
        .supply(2, ("canal-building", "construction"))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "writing", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).hand == (CardId("canal-building"),)
    assert len(result.state.player(P1).hand) == 2


def test_supply_exhaustion_ends_the_game() -> None:
    state = _solo().exhaust_supply(into=P2).build()
    result = resolve_dogma(state, "writing", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert not result.state.pending_effects
