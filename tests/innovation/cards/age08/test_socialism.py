"""SOCIALISM all-or-none tucking and all tied-lowest hand transfers."""

from __future__ import annotations

from support import choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_accepting_tucks_the_complete_hand_and_purple_takes_all_tied_lowest_cards() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("socialism",))
        .hand(P1, ("agriculture", "code-of-laws", "pottery"))
        .hand(P2, ("calendar", "tools", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "socialism",
        choose_branch("tuck-all"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}
    assert result.state.player(P2).hand == (CardId("calendar"),)
    assert result.state.player(P1).board.stack(Color.YELLOW).bottom == CardId("agriculture")
    assert result.state.player(P1).board.stack(Color.BLUE).bottom == CardId("pottery")
    assert result.state.player(P1).board.stack(Color.PURPLE).cards == (
        CardId("code-of-laws"),
        CardId("socialism"),
    )
    tuck_events = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "tuck"
    )
    assert len(tuck_events) == 1
    assert tuck_events[0].change is not None
    assert len(tuck_events[0].change.card_moves) == 3


def test_declining_tucks_none_and_takes_nothing() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("socialism",))
        .hand(P1, ("code-of-laws", "pottery"))
        .hand(P2, ("tools",))
        .build()
    )
    result = resolve_dogma(
        state,
        "socialism",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {CardId("code-of-laws"), CardId("pottery")}
    assert result.state.player(P2).hand == (CardId("tools"),)
