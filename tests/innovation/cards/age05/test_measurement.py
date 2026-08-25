"""MEASUREMENT: optional reveal/return and clarified legal no-op splays."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("measurement",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_the_optional_card_can_be_declined() -> None:
    state = _solo().hand(P1, ("sailing",)).build()
    result = resolve_dogma(state, "measurement", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.qualifying_changes == 0
    assert result.state.player(P1).hand == (CardId("sailing"),)


def test_a_singleton_stack_is_a_legal_no_op_splay_and_still_draws_one() -> None:
    """Official clarification: the player may choose the colour and draw the specified 1."""

    state = _solo().hand(P1, ("sailing",)).supply(1, ("tools",)).build()
    result = resolve_dogma(
        state, "measurement", choose_card("sailing"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.NONE
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert CardId("sailing") in result.state.supply.pile(1)


def test_an_already_right_stack_remains_legal_and_draws_its_card_count() -> None:
    state = (
        scenario(REGISTRY)
        .board(
            P1,
            Color.GREEN,
            ("clothing", "sailing", "measurement"),
            splay=SplayDirection.RIGHT,
        )
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("currency",))
        .supply(3, ("paper",))
        .build()
    )
    result = resolve_dogma(
        state, "measurement", choose_card("currency"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.RIGHT
    assert result.state.player(P1).hand == (CardId("paper"),)


def test_an_absent_matching_colour_draws_from_value_zero_using_age_one() -> None:
    state = _solo().hand(P1, ("tools",)).supply(1, ("writing",)).build()
    result = resolve_dogma(
        state, "measurement", choose_card("tools"), registry=REGISTRY, programs=PROGRAMS
    )
    assert not result.state.player(P1).board.stack(Color.BLUE).cards
    assert result.state.player(P1).hand == (CardId("writing"),)


def test_the_reveal_is_recorded_and_cleared_after_the_return() -> None:
    state = _solo().hand(P1, ("sailing",)).supply(1, ("tools",)).build()
    result = resolve_dogma(
        state,
        "measurement",
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
        verify_resume=True,
    )
    revealed = tuple(
        card_id
        for event in result.events
        if event.kind is EffectEventKind.REVEAL
        for card_id in event.card_ids
    )
    assert revealed == (CardId("sailing"),)
    reveal_return = tuple(event for event in result.events if CardId("sailing") in event.card_ids)
    assert len(reveal_return) == 2
    assert len({event.atomic_group_id for event in reveal_return}) == 1
    assert result.state.revealed == ()


def test_a_stronger_opponent_shares_measurement_first_and_causes_one_bonus_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("measurement",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .hand(P1, ("sailing",))
        .hand(P2, ("city-states",))
        .supply(1, ("tools", "writing"))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "measurement",
        choose_card("city-states"),
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert set(result.state.player(P1).hand) == {CardId("writing"), CardId("physics")}
