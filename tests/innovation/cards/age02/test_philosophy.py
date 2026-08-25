"""PHILOSOPHY: optional dynamic-color splay, legal no-op, scoring, and shared ordinal order."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, choose_color, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("philosophy",))
        .board(P2, Color.RED, ("archery",))
    )


def test_a_present_color_can_be_splayed_then_a_hand_card_scored() -> None:
    state = (
        _solo().board(P1, Color.RED, ("metalworking", "engineering")).hand(P1, ("tools",)).build()
    )
    result = resolve_dogma(
        state,
        "philosophy",
        choose_color(Color.RED),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.LEFT
    assert result.state.player(P1).score_pile == (CardId("tools"),)


def test_both_printed_optional_effects_can_be_declined() -> None:
    state = _solo().hand(P1, ("tools",)).build()
    result = resolve_dogma(
        state,
        "philosophy",
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert result.qualifying_changes == 0


def test_a_singleton_color_is_a_legal_no_op_splay_choice() -> None:
    state = _solo().build()
    result = resolve_dogma(
        state,
        "philosophy",
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.PURPLE).splay is SplayDirection.NONE
    assert result.qualifying_changes == 0


def test_shared_effects_resolve_opponent_then_activator_at_each_ordinal() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("philosophy",))
        .board(P2, Color.PURPLE, ("education",))
        .hand(P1, ("tools",))
        .hand(P2, ("agriculture",))
        .supply(2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(
        state,
        "philosophy",
        decline(),
        decline(),
        choose_card("agriculture"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1, P2, P1)
    assert result.state.player(P2).score_pile == (CardId("agriculture"),)
    assert set(result.state.player(P1).hand) == {
        CardId("tools"),
        CardId("canal-building"),
    }
