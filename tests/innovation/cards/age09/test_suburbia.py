"""SUBURBIA canonical subset, movement order, and fixed reward quantity."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, finish, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.YELLOW, ("suburbia",))


def test_selected_cards_are_ordered_for_tucking_and_each_scores_one_reward() -> None:
    state = _solo().hand(P1, ("tools", "writing")).supply(1, ("agriculture", "clothing")).build()
    result = resolve_dogma(
        state,
        "suburbia",
        choose_card("tools"),
        choose_card("writing"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.BLUE).cards == (
        CardId("tools"),
        CardId("writing"),
    )
    assert set(result.state.player(P1).score_pile) == {
        CardId("agriculture"),
        CardId("clothing"),
    }


def test_finishing_with_no_selection_draws_no_rewards() -> None:
    state = _solo().hand(P1, ("tools",)).supply(1, ("agriculture",)).build()
    result = resolve_dogma(
        state,
        "suburbia",
        finish(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert not result.state.player(P1).score_pile
