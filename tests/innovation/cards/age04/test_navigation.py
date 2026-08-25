"""NAVIGATION: filtered private score choice, hidden observations, no-op, and immunity."""

from __future__ import annotations

from support import ScenarioBuilder, assert_no_leak, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs, start_dogma
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("navigation",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_the_victim_chooses_an_exact_value_two_or_three_score_card() -> None:
    state = _vulnerable().score(P2, ("tools", "calendar", "paper", "anatomy")).build()
    result = resolve_dogma(
        state,
        "navigation",
        choose_card("paper"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    decision = result.decisions[0]
    assert decision.chooser is P2
    assert {
        action.card_id for action in decision.legal_actions if isinstance(action, ChooseCardAction)
    } == {CardId("calendar"), CardId("paper")}
    assert result.state.player(P1).score_pile == (CardId("paper"),)
    assert set(result.state.player(P2).score_pile) == {
        CardId("tools"),
        CardId("calendar"),
        CardId("anatomy"),
    }


def test_no_value_two_or_three_score_card_makes_the_demand_a_no_op() -> None:
    state = _vulnerable().score(P2, ("tools", "anatomy")).build()
    result = resolve_dogma(state, "navigation", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert set(result.state.player(P2).score_pile) == {CardId("tools"), CardId("anatomy")}


def test_equal_crown_counts_make_the_opponent_immune() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("navigation",))
        .board(P2, Color.PURPLE, ("enterprise",))
        .score(P2, ("calendar",))
        .build()
    )
    result = resolve_dogma(state, "navigation", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).score_pile == (CardId("calendar"),)


def test_swapping_equal_value_private_identities_does_not_leak_to_the_activator() -> None:
    first = _vulnerable().score(P2, ("calendar", "paper")).hand(P2, ("canal-building",)).build()
    second = _vulnerable().score(P2, ("canal-building", "paper")).hand(P2, ("calendar",)).build()
    first_paused = start_dogma(first, CardId("navigation"), P1, PROGRAMS, REGISTRY)
    second_paused = start_dogma(second, CardId("navigation"), P1, PROGRAMS, REGISTRY)
    assert first_paused.decision is not None and first_paused.decision.chooser is P2
    assert second_paused.decision is not None and second_paused.decision.chooser is P2
    assert_no_leak(first_paused.state, second_paused.state, P1, REGISTRY)
