"""EDUCATION: optional tied-highest return and live highest-plus-two draw value."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("education",))
        .board(P2, Color.GREEN, ("the-wheel",))
    )


def test_the_optional_return_can_be_declined() -> None:
    state = _solo().score(P1, ("machinery",)).build()
    result = resolve_dogma(state, "education", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("machinery"),)
    assert result.state.player(P1).hand == ()


def test_the_owner_breaks_a_tied_highest_and_draws_from_the_remaining_highest() -> None:
    state = _solo().score(P1, ("compass", "machinery", "tools")).supply(5, ("astronomy",)).build()
    result = resolve_dogma(
        state,
        "education",
        choose_card("compass"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("compass"), CardId("machinery")}
    assert result.state.player(P1).score_pile == (CardId("machinery"), CardId("tools"))
    assert result.state.player(P1).hand == (CardId("astronomy"),)


def test_returning_the_only_score_card_draws_a_two_from_absent_value_zero_plus_two() -> None:
    state = _solo().score(P1, ("tools",)).supply(2, ("calendar",)).build()
    result = resolve_dogma(
        state,
        "education",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == ()
    assert result.state.player(P1).hand == (CardId("calendar"),)
