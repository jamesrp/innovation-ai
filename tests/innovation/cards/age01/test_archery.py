"""ARCHERY: demand pronouns, immunity at equal icons, and a hidden-hand tie."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs, start_dogma
from innovation_ai.innovation.effects.dogma import dogma_schedule
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    """Return a position where the opponent has strictly fewer castles than Archery's two."""

    return scenario(REGISTRY).board(P1, Color.RED, ("archery",)).board(P2, Color.BLUE, ("pottery",))


def test_the_demand_draws_for_the_victim_then_transfers_to_the_demander() -> None:
    """Rule 8.3 pronouns: "your hand" is the victim's, "my hand" is the activator's."""

    state = _vulnerable().hand(P2, ("tools",)).supply(1, ("writing",)).build()
    result = resolve_dogma(
        state, "archery", choose_card("writing"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.status is EffectStatus.COMPLETE
    # Both drawn writing and held tools are value 1, so either is a legal highest choice.
    assert result.state.player(P1).hand == (CardId("writing"),)
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert_conserved(result.state, REGISTRY)


def test_an_equal_icon_count_grants_complete_immunity() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.RED, ("oars",))
        .hand(P2, ("tools",))
        .build()
    )
    started = start_dogma(
        state, CardId("archery"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    assert dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS) == ()
    result = resolve_dogma(state, "archery", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert not result.state.player(P1).hand
    assert result.qualifying_changes == 0


def test_a_stronger_opponent_is_also_immune() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "archery", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).hand == (CardId("tools"),)


def test_the_victim_chooses_among_tied_highest_cards_in_their_own_hand() -> None:
    """Decision 13: the zone owner picks the exact identity among equal-value candidates."""

    state = (
        _vulnerable().hand(P2, ("canal-building", "construction")).supply(1, ("writing",)).build()
    )
    result = resolve_dogma(
        state, "archery", choose_card("construction"), registry=REGISTRY, programs=PROGRAMS
    )
    decision = result.decisions[0]
    # The victim chooses, not the demander who cannot see the hand.
    assert decision.chooser is P2
    assert decision.executor is P2
    assert decision.context is not None and decision.context.demand
    offered = {action.card_id for action in decision.legal_actions if hasattr(action, "card_id")}
    # Both age 2 cards are highest; the drawn age 1 writing is not.
    assert offered == {CardId("canal-building"), CardId("construction")}
    assert result.state.player(P1).hand == (CardId("construction"),)


def test_partial_execution_transfers_only_the_drawn_card_from_an_empty_hand() -> None:
    """Rule 8.5: an empty hand still receives the draw, which then becomes the highest card."""

    state = _vulnerable().supply(1, ("writing",)).build()
    result = resolve_dogma(state, "archery", registry=REGISTRY, programs=PROGRAMS)
    # Exactly one candidate, so decision 13's fallback resolves it with no decision.
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("writing"),)
    assert not result.state.player(P2).hand


def test_a_demand_never_earns_the_activator_a_free_draw() -> None:
    state = _vulnerable().hand(P2, ("tools",)).supply(1, ("writing",)).build()
    result = resolve_dogma(
        state, "archery", choose_card("tools"), registry=REGISTRY, programs=PROGRAMS
    )
    assert len(result.state.player(P1).hand) == 1


def test_the_demand_events_carry_full_demand_provenance() -> None:
    state = _vulnerable().hand(P2, ("tools",)).supply(1, ("writing",)).build()
    result = resolve_dogma(
        state, "archery", choose_card("tools"), registry=REGISTRY, programs=PROGRAMS
    )
    for event in result.events:
        if not event.changed:
            continue
        assert event.demand
        assert not event.shared
        assert event.executor is P2
        assert event.dogma_activator is P1
        assert event.source_card_id == CardId("archery")
