"""GUNPOWDER: victim top choice, demand-caused follow-up, no-op, and immunity."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
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
        .board(P1, Color.RED, ("gunpowder",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_the_victim_chooses_one_castle_top_and_the_activator_scores_a_two() -> None:
    state = _vulnerable().board(P2, Color.YELLOW, ("masonry",)).supply(2, ("calendar",)).build()
    result = resolve_dogma(
        state,
        "gunpowder",
        choose_card("masonry"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseCardAction)
    }
    assert offered == {CardId("metalworking"), CardId("masonry")}
    assert result.decisions[0].chooser is P2
    assert set(result.state.player(P1).score_pile) == {
        CardId("masonry"),
        CardId("calendar"),
    }
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("metalworking")


def test_no_castle_top_skips_the_follow_up_draw_and_score() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("gunpowder",))
        .board(P2, Color.PURPLE, ("education",))
        .supply(2, ("calendar",))
        .build()
    )
    result = resolve_dogma(state, "gunpowder", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert not result.state.player(P1).score_pile
    assert CardId("calendar") in result.state.supply.pile(2)


def test_a_single_castle_target_still_records_the_victims_choice() -> None:
    state = _vulnerable().supply(2, ("calendar",)).build()
    result = resolve_dogma(
        state,
        "gunpowder",
        choose_card("metalworking"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert CardId("metalworking") in result.state.player(P1).score_pile
    assert CardId("calendar") in result.state.player(P1).score_pile


def test_a_stronger_factory_opponent_is_immune_and_no_follow_up_occurs() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("gunpowder",))
        .board(P2, Color.RED, ("coal",))
        .board(P2, Color.YELLOW, ("masonry",))
        .supply(2, ("calendar",))
        .build()
    )
    result = resolve_dogma(state, "gunpowder", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P2).board.stack(Color.YELLOW).top == CardId("masonry")
    assert not result.state.player(P1).score_pile
