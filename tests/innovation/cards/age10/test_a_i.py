"""A.I. split-board prerequisite and unique-lowest-score win."""

from __future__ import annotations

from support import ScenarioBuilder, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _split_required_cards() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("a-i",))
        .board(P1, Color.RED, ("robotics",))
        .board(P2, Color.BLUE, ("software",))
        .supply(10, ("databases",))
    )


def test_robotics_and_software_may_be_top_cards_on_different_boards() -> None:
    state = _split_required_cards().score(P2, ("tools",)).build()
    result = resolve_dogma(
        state,
        "a-i",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert result.terminal.winners == (P2,)


def test_a_lowest_score_tie_ignores_the_win_effect() -> None:
    state = _split_required_cards().score(P2, ("globalization",)).build()
    result = resolve_dogma(
        state,
        "a-i",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.terminal is None
