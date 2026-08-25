"""PRINTING PRESS: optional score return, purple-relative draw, splay, and sharing."""

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
        .board(P1, Color.BLUE, ("printing-press",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_a_return_draws_two_above_the_top_purple_cards_value() -> None:
    state = (
        _solo()
        .board(P1, Color.PURPLE, ("education",))
        .score(P1, ("tools",))
        .supply(5, ("banking",))
        .build()
    )
    result = resolve_dogma(
        state,
        "printing-press",
        choose_card("tools"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("banking"),)
    assert not result.state.player(P1).score_pile


def test_no_purple_stack_means_the_return_draws_a_two() -> None:
    state = _solo().score(P1, ("tools",)).supply(2, ("calendar",)).build()
    result = resolve_dogma(
        state,
        "printing-press",
        choose_card("tools"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("calendar"),)


def test_the_score_return_can_be_declined_while_blue_is_splayed_right() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools", "printing-press"))
        .board(P2, Color.RED, ("metalworking",))
        .score(P1, ("calendar",))
        .build()
    )
    result = resolve_dogma(
        state,
        "printing-press",
        decline(),
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).score_pile == (CardId("calendar"),)
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.RIGHT


def test_a_stronger_bulb_opponent_shares_both_effects_before_the_activator() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("tools", "printing-press"))
        .board(P2, Color.BLUE, ("experimentation",))
        .score(P1, ("writing",))
        .score(P2, ("sailing",))
        .supply(2, ("calendar", "canal-building"))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "printing-press",
        choose_card("sailing"),
        choose_card("writing"),
        choose_color(Color.BLUE),
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1, P2, P1)
    assert result.state.player(P2).hand == (CardId("calendar"),)
    assert set(result.state.player(P1).hand) == {CardId("canal-building"), CardId("anatomy")}
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.RIGHT
