"""ECOLOGY optional return and conditional reward."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.YELLOW, ("ecology",))


def test_return_scores_another_hand_card_and_draws_two_tens() -> None:
    state = _solo().hand(P1, ("tools", "writing")).supply(10, ("databases", "robotics")).build()
    result = resolve_dogma(
        state,
        "ecology",
        choose_card("tools"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("writing"),)
    assert set(result.state.player(P1).hand) == {CardId("databases"), CardId("robotics")}
    assert result.state.supply.pile(1)[-1] == CardId("tools")


def test_declining_the_return_skips_the_entire_reward() -> None:
    state = _solo().hand(P1, ("tools",)).supply(10, ("databases", "robotics")).build()
    result = resolve_dogma(
        state,
        "ecology",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert not result.state.player(P1).score_pile
