"""THE PIRATE CODE: victim-owned subset choices, partial demand, and crown tie choices."""

from __future__ import annotations

from support import ScenarioBuilder, assert_no_leak, choose_card, resolve_dogma, scenario

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
        .board(P1, Color.RED, ("the-pirate-code",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_the_victim_chooses_exactly_two_eligible_private_score_cards() -> None:
    state = (
        _vulnerable()
        .board(P1, Color.GREEN, ("sailing",))
        .score(P2, ("tools", "calendar", "anatomy", "astronomy"))
        .build()
    )
    result = resolve_dogma(
        state,
        "the-pirate-code",
        choose_card("anatomy"),
        choose_card("tools"),
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2, P1)
    assert set(result.state.player(P1).score_pile) == {
        CardId("anatomy"),
        CardId("tools"),
        CardId("sailing"),
    }
    assert set(result.state.player(P2).score_pile) == {CardId("calendar"), CardId("astronomy")}


def test_one_eligible_card_is_transferred_under_mandatory_partial_execution() -> None:
    state = (
        _vulnerable().board(P1, Color.GREEN, ("sailing",)).score(P2, ("tools", "astronomy")).build()
    )
    result = resolve_dogma(
        state,
        "the-pirate-code",
        choose_card("tools"),
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert CardId("tools") in result.state.player(P1).score_pile


def test_no_eligible_score_card_skips_both_the_transfer_and_follow_up() -> None:
    state = _vulnerable().score(P2, ("astronomy",)).build()
    result = resolve_dogma(state, "the-pirate-code", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).score_pile == (CardId("astronomy"),)
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("the-pirate-code")


def test_the_activator_breaks_a_tie_between_lowest_crown_tops() -> None:
    state = (
        _vulnerable()
        .board(P1, Color.GREEN, ("sailing",))
        .board(P1, Color.PURPLE, ("city-states",))
        .score(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(
        state,
        "the-pirate-code",
        choose_card("tools"),
        choose_card("city-states"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id
        for action in result.decisions[-1].legal_actions
        if hasattr(action, "card_id")
    }
    assert offered == {CardId("sailing"), CardId("city-states")}
    assert CardId("city-states") in result.state.player(P1).score_pile


def test_private_victim_options_do_not_leak_to_the_demander() -> None:
    first = _vulnerable().score(P2, ("anatomy", "enterprise")).hand(P2, ("gunpowder",)).build()
    second = _vulnerable().score(P2, ("anatomy", "gunpowder")).hand(P2, ("enterprise",)).build()
    first_paused = start_dogma(first, CardId("the-pirate-code"), P1, PROGRAMS, REGISTRY)
    second_paused = start_dogma(second, CardId("the-pirate-code"), P1, PROGRAMS, REGISTRY)
    assert first_paused.decision is not None and first_paused.decision.chooser is P2
    assert second_paused.decision is not None and second_paused.decision.chooser is P2
    assert_no_leak(first_paused.state, second_paused.state, P1, REGISTRY)


def test_demand_immunity_means_the_conditional_follow_up_does_nothing() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("the-pirate-code",))
        .board(P2, Color.PURPLE, ("enterprise",))
        .score(P2, ("tools", "calendar"))
        .build()
    )
    result = resolve_dogma(state, "the-pirate-code", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert set(result.state.player(P2).score_pile) == {CardId("tools"), CardId("calendar")}
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("the-pirate-code")
