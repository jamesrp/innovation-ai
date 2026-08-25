"""SOCIETIES: authoritative top-card scope and same-colour value comparison."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("societies",))
        .board(P2, Color.YELLOW, ("agriculture",))
    )


def test_absent_same_colour_top_has_value_zero_and_the_victim_draws_after_transfer() -> None:
    state = _vulnerable().board(P2, Color.BLUE, ("tools",)).supply(5, ("physics",)).build()
    result = resolve_dogma(
        state, "societies", choose_card("tools"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.decisions[0].chooser is P2
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("tools")
    assert result.state.player(P2).hand == (CardId("physics"),)


def test_a_covered_bulb_card_is_excluded_by_the_top_card_wording() -> None:
    state = (
        _vulnerable()
        .board(P2, Color.BLUE, ("tools", "translation"))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(state, "societies", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.BLUE).cards == (
        CardId("tools"),
        CardId("translation"),
    )
    assert not result.state.player(P2).hand


def test_only_bulb_tops_higher_than_the_activators_same_colour_top_are_offered() -> None:
    state = (
        _vulnerable()
        .board(P1, Color.BLUE, ("chemistry",))
        .board(P1, Color.GREEN, ("banking",))
        .board(P2, Color.BLUE, ("atomic-theory",))
        .board(P2, Color.GREEN, ("paper",))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(
        state,
        "societies",
        choose_card("atomic-theory"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("atomic-theory")}


def test_equal_crown_count_grants_complete_demand_immunity() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("societies",))
        .board(P2, Color.PURPLE, ("enterprise",))
        .board(P2, Color.BLUE, ("atomic-theory",))
        .supply(5, ("physics",))
        .build()
    )
    result = resolve_dogma(state, "societies", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.BLUE).top == CardId("atomic-theory")
    assert not result.state.player(P2).hand
