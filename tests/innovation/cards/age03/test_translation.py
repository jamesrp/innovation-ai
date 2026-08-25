"""TRANSLATION: all-or-none score melding, movement order, and the World linked route."""

from __future__ import annotations

from support import choose_branch, choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectStatus,
    load_effect_programs,
    start_effect,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId, SpecialAchievementId
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_accepting_the_option_melds_the_complete_score_pile_as_one_atom() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("translation",))
        .board(P2, Color.RED, ("archery",))
        .score(P1, ("sailing", "compass"))
        .build()
    )
    result = resolve_dogma(
        state,
        "translation",
        choose_branch("meld-all"),
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == ()
    assert result.state.player(P1).board.stack(Color.GREEN).cards == (
        CardId("sailing"),
        CardId("compass"),
    )
    melds = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.MELD
    )
    assert len(melds) == 1 and set(melds[0].card_ids) == {
        CardId("sailing"),
        CardId("compass"),
    }
    # There is an optional all/none branch and an order decision, never a subset selection.
    assert len(result.decisions) == 2
    assert SpecialAchievementId.WORLD in result.state.player(P1).special_achievements


def test_declining_melds_none_and_a_non_crown_top_blocks_world() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("translation",))
        .board(P1, Color.RED, ("metalworking",))
        .board(P2, Color.YELLOW, ("machinery",))
        .score(P1, ("sailing", "compass"))
        .build()
    )
    result = resolve_dogma(
        state,
        "translation",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).score_pile) == {CardId("sailing"), CardId("compass")}
    assert SpecialAchievementId.WORLD not in result.state.player(P1).special_achievements


def test_empty_score_needs_no_meld_choice_and_empty_top_set_satisfies_world_vacuously() -> None:
    state = scenario(REGISTRY).build()
    context = EffectContext(
        actor=P1,
        chooser=P1,
        executor=P1,
        dogma_activator=P1,
        source_card_id=CardId("translation"),
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
    )
    result = start_effect(state, "translation-v1", context, PROGRAMS, REGISTRY)
    assert result.status is EffectStatus.COMPLETE
    assert SpecialAchievementId.WORLD in result.state.player(P1).special_achievements
