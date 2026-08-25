"""MACHINE TOOLS: highest-score quantity, value-zero draw, and immediate terminal draw."""

from __future__ import annotations

from support import resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.state import TerminalReason
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_empty_score_pile_requests_zero_and_draws_then_scores_an_age_one() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("machine-tools",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .supply(1, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "machine-tools", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("tools"),)
    events = tuple(event for event in result.events if event.change is not None)
    assert len(events) == 2
    assert len({event.atomic_group_id for event in events}) == 1


def test_exhausted_age_ten_request_ends_the_game_immediately() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("machine-tools",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .score(P1, ("software",))
        .exhaust_supply(into=P1)
        .build()
    )
    result = resolve_dogma(state, "machine-tools", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.reason is TerminalReason.DRAW_BEYOND_AGE_10
    assert result.terminal.winners == (P1,)
