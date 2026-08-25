"""COLONIALISM: crown-driven repeat, atomic tucks, sharing, and deterministic replay."""

from __future__ import annotations

from support import ScenarioBuilder, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.state import state_hash
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("colonialism",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_crown_draws_are_tucked_before_the_effect_repeats() -> None:
    state = _solo().supply(3, ("compass", "paper", "engineering")).build()
    result = resolve_dogma(state, "colonialism", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).board.stack(Color.GREEN).cards == (
        CardId("paper"),
        CardId("compass"),
    )
    assert result.state.player(P1).board.stack(Color.RED).bottom == CardId("engineering")
    assert not result.state.player(P1).hand


def test_a_non_crown_draw_stops_after_one_atomic_draw_and_tuck() -> None:
    state = _solo().supply(3, ("alchemy", "compass")).build()
    result = resolve_dogma(state, "colonialism", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).board.stack(Color.BLUE).bottom == CardId("alchemy")
    assert CardId("compass") in result.state.supply.pile(3)
    groups = {
        event.atomic_group_id for event in result.events if CardId("alchemy") in event.card_ids
    }
    assert len(groups) == 1


def test_a_stronger_factory_opponent_shares_first_and_earns_a_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("colonialism",))
        .board(P2, Color.RED, ("coal",))
        .supply(3, ("engineering", "alchemy"))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(state, "colonialism", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P2).board.stack(Color.RED).bottom == CardId("engineering")
    assert result.state.player(P1).board.stack(Color.BLUE).bottom == CardId("alchemy")
    assert result.state.player(P1).hand == (CardId("anatomy"),)


def test_the_repeat_path_replays_identically_across_serialized_boundaries() -> None:
    state = _solo().supply(3, ("compass", "engineering")).build()
    first = resolve_dogma(
        state,
        "colonialism",
        registry=REGISTRY,
        programs=PROGRAMS,
        verify_resume=True,
    )
    second = resolve_dogma(
        state,
        "colonialism",
        registry=REGISTRY,
        programs=PROGRAMS,
        verify_resume=True,
    )
    assert state_hash(first.state) == state_hash(second.state)
    assert first.events == second.events
