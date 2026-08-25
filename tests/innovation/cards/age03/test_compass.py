"""COMPASS: victim-chosen public tops, live second selector, and partial execution."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_the_victim_transfers_each_qualifying_top_card_between_boards() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("compass",))
        .board(P1, Color.PURPLE, ("education",))
        .board(P2, Color.YELLOW, ("machinery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "compass",
        choose_card("machinery"),
        choose_card("education"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("machinery")
    assert result.state.player(P2).board.stack(Color.PURPLE).top == CardId("education")
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2)


def test_the_second_transfer_still_occurs_when_the_first_has_no_candidate() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("compass",))
        .board(P1, Color.PURPLE, ("education",))
        # Green is excluded from the first clause, and Paper has no leaf anyway.
        .board(P2, Color.GREEN, ("paper",))
        .build()
    )
    result = resolve_dogma(
        state,
        "compass",
        choose_card("education"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert result.state.player(P2).board.stack(Color.PURPLE).top == CardId("education")


def test_a_stronger_crown_count_makes_the_opponent_immune_to_the_entire_demand() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("compass",))
        .board(P1, Color.PURPLE, ("education",))
        .board(P2, Color.BLUE, ("translation",))
        .board(P2, Color.YELLOW, ("machinery",))
        .build()
    )
    result = resolve_dogma(state, "compass", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.YELLOW).top == CardId("machinery")
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("education")
