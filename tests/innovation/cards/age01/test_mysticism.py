"""MYSTICISM: reveal branches, physical reveal cleanup, extra draw, and reveal-only sharing."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, load_effect_programs
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("mysticism",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_a_nonmatching_card_is_kept_and_is_no_longer_public() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "mysticism", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("agriculture"),)
    assert result.state.revealed == ()
    assert observe(result.state, P2, REGISTRY).player(P1).hand.known_cards == ()
    assert any(event.kind is EffectEventKind.KEEP for event in result.events)


def test_a_matching_card_is_melded_then_an_extra_one_is_drawn() -> None:
    state = _solo().supply(1, ("city-states", "agriculture")).build()
    result = resolve_dogma(state, "mysticism", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("city-states")
    assert result.state.player(P1).hand == (CardId("agriculture"),)
    assert result.state.revealed == ()
    assert_conserved(result.state, REGISTRY)


def test_each_initial_draw_is_revealed_exactly_once_and_atomically() -> None:
    state = _solo().supply(1, ("city-states", "agriculture")).build()
    result = resolve_dogma(state, "mysticism", registry=REGISTRY, programs=PROGRAMS)
    revealed = tuple(
        card_id
        for event in result.events
        if event.kind is EffectEventKind.REVEAL
        for card_id in event.card_ids
    )
    assert revealed == (CardId("city-states"),)
    draw_reveal = tuple(
        event
        for event in result.events
        if CardId("city-states") in event.card_ids
        and (
            event.kind is EffectEventKind.REVEAL
            or (event.change is not None and event.change.kind.value == "draw")
        )
    )
    assert len(draw_reveal) == 2
    assert len({event.atomic_group_id for event in draw_reveal}) == 1


def test_a_shared_nonmatching_reveal_still_earns_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("mysticism",))
        .board(P2, Color.RED, ("metalworking",))
        .supply(1, ("agriculture", "clothing", "domestication"))
        .build()
    )
    result = resolve_dogma(state, "mysticism", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P2).hand) == 1
    assert len(result.state.player(P1).hand) == 2
    assert result.state.revealed == ()
