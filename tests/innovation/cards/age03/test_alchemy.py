"""ALCHEMY: reveal tracking, red branching, atomic whole-hand return, and effect two."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _four_castles() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("alchemy",))
        .board(P1, Color.RED, ("engineering",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_non_red_draw_is_revealed_kept_then_available_to_meld_and_score() -> None:
    state = _four_castles().hand(P1, ("tools",)).supply(4, ("enterprise",)).build()
    result = resolve_dogma(
        state,
        "alchemy",
        choose_card("enterprise"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("enterprise")
    assert result.state.player(P1).score_pile == (CardId("tools"),)
    reveals = tuple(event for event in result.events if event.kind is EffectEventKind.REVEAL)
    assert tuple(card for event in reveals for card in event.card_ids) == (CardId("enterprise"),)
    assert result.state.revealed == ()


def test_a_red_draw_returns_the_drawn_card_and_the_preexisting_hand_in_one_atom() -> None:
    state = _four_castles().hand(P1, ("tools",)).supply(4, ("gunpowder",)).build()
    result = resolve_dogma(state, "alchemy", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == ()
    returns = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.RETURN
    )
    assert len(returns) == 1
    assert set(returns[0].card_ids) == {CardId("gunpowder"), CardId("tools")}
    assert len(returns[0].change.card_moves) == 2  # type: ignore[union-attr]
    assert result.state.revealed == ()


def test_all_draws_remain_public_until_the_batch_return_order_choice() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("alchemy",))
        .board(P1, Color.RED, ("engineering",))
        .board(P1, Color.PURPLE, ("feudalism",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(4, ("gunpowder", "enterprise"))
        .build()
    )
    result = resolve_dogma(
        state,
        "alchemy",
        # The two age-4 returns share a pile, so their owner chooses their order.
        choose_card("enterprise"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    revealed = tuple(
        card
        for event in result.events
        if event.kind is EffectEventKind.REVEAL
        for card in event.card_ids
    )
    assert revealed == (CardId("gunpowder"), CardId("enterprise"))
    assert len(result.decisions) == 1
    assert set(result.decisions[0].observation.revealed_cards) == set(revealed)
    returns = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.RETURN
    )
    assert len(returns) == 1 and set(returns[0].card_ids) == set(revealed)
