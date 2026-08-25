"""OARS: repeated hidden-hand demand, fallback history, immunity sharing, and interruption."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.RED, ("oars",)).board(P2, Color.BLUE, ("pottery",))


def test_no_crown_transfer_causes_the_activators_fallback_draw() -> None:
    state = _vulnerable().hand(P2, ("tools",)).supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "oars", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert result.state.player(P1).hand == (CardId("agriculture"),)


def test_each_successful_transfer_draws_for_the_victim_and_repeats() -> None:
    state = (
        _vulnerable()
        .hand(P2, ("city-states", "clothing"))
        .supply(1, ("agriculture", "tools"))
        .build()
    )
    result = resolve_dogma(
        state, "oars", choose_card("city-states"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.decisions[0].chooser is P2
    assert set(result.state.player(P1).score_pile) == {
        CardId("city-states"),
        CardId("clothing"),
    }
    assert set(result.state.player(P2).hand) == {CardId("agriculture"), CardId("tools")}
    assert not result.state.player(P1).hand, "a successful demand suppresses effect two"
    assert_conserved(result.state, REGISTRY)


def test_transfers_never_count_as_sharing_credit() -> None:
    state = _vulnerable().hand(P2, ("city-states",)).supply(1, ("tools",)).build()
    result = resolve_dogma(state, "oars", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).score_pile == (CardId("city-states"),)
    assert not result.state.player(P1).hand


def test_an_immune_opponent_shares_effect_two_and_both_players_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("oars",))
        .board(P2, Color.RED, ("metalworking",))
        .supply(1, ("agriculture", "clothing", "domestication"))
        .build()
    )
    result = resolve_dogma(state, "oars", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P2).hand) == 1
    # Own effect-two draw plus the one sharing-bonus draw.
    assert len(result.state.player(P1).hand) == 2


def test_terminal_during_the_demand_skips_the_fallback_effect() -> None:
    state = _vulnerable().hand(P2, ("city-states",)).exhaust_supply(into=P1).build()
    result = resolve_dogma(state, "oars", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert CardId("city-states") in result.state.player(P1).score_pile
