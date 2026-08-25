"""ROBOTICS top-green score and nested execution of the melded 10."""

from __future__ import annotations

from support import resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_the_melded_ten_executes_its_non_demand_effects_nested() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("robotics",))
        .board(P1, Color.GREEN, ("self-service",))
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(10, ("globalization",))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(state, "robotics", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("globalization")
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("code-of-laws")
    assert set(result.state.player(P1).score_pile) == {
        CardId("self-service"),
        CardId("canning"),
    }
    nested = tuple(
        event for event in result.events if event.source_card_id == CardId("globalization")
    )
    assert nested and all(
        event.nested and not event.demand and not event.shared for event in nested
    )
