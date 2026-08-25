"""REFORMATION: leaf-snapshotted optional tucks, early stop, and right splay."""

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
        .board(P1, Color.PURPLE, ("reformation",))
        .board(P2, Color.BLUE, ("experimentation",))
    )


def test_the_sources_three_leaves_allow_one_optional_tuck() -> None:
    state = _solo().hand(P1, ("city-states", "tools")).build()
    result = resolve_dogma(
        state,
        "reformation",
        choose_card("city-states"),
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert result.state.player(P1).board.stack(Color.PURPLE).bottom == CardId("city-states")
    assert result.state.player(P1).board.stack(Color.PURPLE).splay is SplayDirection.RIGHT


def test_declining_one_tuck_stops_tucking_but_not_the_next_printed_effect() -> None:
    state = _solo().hand(P1, ("tools", "writing")).build()
    result = resolve_dogma(
        state,
        "reformation",
        decline(),
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}
    assert result.state.player(P1).board.stack(Color.PURPLE).splay is SplayDirection.NONE
    assert len(result.decisions) == 2


def test_tuck_quantity_is_snapshotted_before_new_leaf_stacks_appear() -> None:
    state = (
        _solo()
        .board(P1, Color.YELLOW, ("agriculture",))
        .hand(P1, ("pottery", "clothing", "machinery", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "reformation",
        choose_card("pottery"),
        choose_card("clothing"),
        choose_card("machinery"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    # Six initial leaves permit exactly three tucks. Pottery and Clothing add visible leaves,
    # but decision 17 does not grow this instruction's already-snapshotted quantity.
    assert result.state.player(P1).hand == (CardId("writing"),)
    assert len(result.decisions) == 4  # three tucks, then the optional splay decline


def test_only_existing_yellow_or_purple_stacks_are_splay_options() -> None:
    state = _solo().hand(P1, ()).build()
    result = resolve_dogma(
        state,
        "reformation",
        choose_color(Color.PURPLE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.color for action in result.decisions[0].legal_actions if hasattr(action, "color")
    }
    assert offered == {Color.PURPLE}
