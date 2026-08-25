"""SAILING: atomic draw-and-meld, sharing, upward fallback, and terminal unwind."""

from __future__ import annotations

from support import assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo():  # type: ignore[no-untyped-def]
    return (
        scenario(REGISTRY).board(P1, Color.GREEN, ("sailing",)).board(P2, Color.RED, ("archery",))
    )


def test_draw_and_meld_uses_the_exact_drawn_card() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "sailing", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("agriculture")
    assert not result.state.player(P1).hand
    assert_conserved(result.state, REGISTRY)


def test_the_draw_and_meld_changes_share_one_atomic_group() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "sailing", registry=REGISTRY, programs=PROGRAMS)
    changed = tuple(event for event in result.events if event.changed)
    assert len(changed) == 2
    assert len({event.atomic_group_id for event in changed}) == 1


def test_empty_age_one_uses_the_next_nonempty_pile() -> None:
    age_one = tuple(
        sorted(
            card.id
            for card in REGISTRY.cards
            if card.age == 1 and card.id not in {CardId("sailing"), CardId("archery")}
        )
    )
    state = _solo().score(P2, age_one[1:]).supply(2, ("canal-building",)).build()
    assert not state.supply.pile(1)
    result = resolve_dogma(state, "sailing", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("canal-building")


def test_equal_crowns_share_then_award_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("sailing",))
        .board(P2, Color.PURPLE, ("city-states",))
        .supply(1, ("agriculture", "clothing", "domestication"))
        .build()
    )
    result = resolve_dogma(state, "sailing", registry=REGISTRY, programs=PROGRAMS)
    assert sum(len(stack.cards) for stack in result.state.player(P2).board.stacks) == 2
    assert sum(len(stack.cards) for stack in result.state.player(P1).board.stacks) == 2
    assert len(result.state.player(P1).hand) == 1


def test_an_impossible_draw_terminates_before_any_meld() -> None:
    state = _solo().exhaust_supply(into=P2).build()
    result = resolve_dogma(state, "sailing", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
