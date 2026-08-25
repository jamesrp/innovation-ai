"""SOFTWARE scores one 10, then nests only into the second of two melded 10s."""

from __future__ import annotations

from support import resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_only_the_second_melded_ten_executes_nested() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("software",))
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(10, ("a-i", "databases", "globalization"))
        .supply(6, ("canning",))
        .build()
    )
    result = resolve_dogma(state, "software", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).score_pile) == {
        CardId("a-i"),
        CardId("canning"),
    }
    assert result.state.player(P1).board.stack(Color.GREEN).top == CardId("databases")
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("globalization")
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("code-of-laws")
    nested_events = tuple(event for event in result.events if event.nested and event.changed)
    assert {event.source_card_id for event in nested_events} == {CardId("globalization")}
    assert all(not event.demand and not event.shared for event in nested_events)
