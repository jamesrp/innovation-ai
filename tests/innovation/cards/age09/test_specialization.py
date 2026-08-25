"""SPECIALIZATION reveal-derived color transfer and restricted splay choice."""

from __future__ import annotations

from support import choose_card, choose_color, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseColorAction, DeclineAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_revealed_hand_color_takes_only_the_opponents_top_card_of_that_color() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("specialization",))
        .hand(P1, ("tools",))
        .board(P2, Color.BLUE, ("pottery", "calendar"))
        .build()
    )
    result = resolve_dogma(
        state,
        "specialization",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("calendar")}
    assert result.state.player(P2).board.stack(Color.BLUE).cards == (CardId("pottery"),)
    assert any(event.kind is EffectEventKind.REVEAL for event in result.events)
    assert not result.state.revealed


def test_empty_hand_partially_executes_without_reveal_or_transfer() -> None:
    state = scenario(REGISTRY).board(P1, Color.PURPLE, ("specialization",)).build()
    result = resolve_dogma(state, "specialization", registry=REGISTRY, programs=PROGRAMS)

    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.events == ()
    assert result.qualifying_changes == 0
    assert not result.state.revealed


def test_shared_executor_can_reveal_before_empty_activator_execution() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("specialization",))
        .board(P2, Color.RED, ("coal",))
        .hand(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(
        state,
        "specialization",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    reveals = tuple(event for event in result.events if event.kind is EffectEventKind.REVEAL)
    assert len(reveals) == 1
    assert reveals[0].executor is P2
    assert all(
        event.executor is not P1
        or event.source_effect_id is None
        or event.source_effect_id.ordinal != 1
        for event in result.events
    )
    assert not result.state.revealed


def test_optional_splay_offers_present_yellow_but_not_absent_blue() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("specialization",))
        .board(P1, Color.YELLOW, ("ecology",))
        .hand(P1, ("tools",))
        .build()
    )
    result = resolve_dogma(
        state,
        "specialization",
        choose_card("tools"),
        choose_color(Color.YELLOW),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    splay_decision = result.decisions[-1]
    assert any(isinstance(action, DeclineAction) for action in splay_decision.legal_actions)
    assert {
        action.color
        for action in splay_decision.legal_actions
        if isinstance(action, ChooseColorAction)
    } == {Color.YELLOW}
    assert result.state.player(P1).board.stack(Color.YELLOW).splay is SplayDirection.NONE
