"""GUNPOWDER: victim top choice, demand-caused follow-up, no-op, and immunity."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectProgramRegistry,
    EffectStatus,
    load_effect_programs,
    start_effect,
    submit_effect_action,
)
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    NestedNode,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, PlayerId

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


def test_nested_gunpowder_ignores_unrelated_changes_in_the_outer_dogma() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("self-service",))
        .board(P1, Color.RED, ("gunpowder",))
        .board(P2, Color.RED, ("optics",))
        .board(P2, Color.BLUE, ("experimentation",))
        .board(P2, Color.GREEN, ("currency",))
        .supply(2, ("calendar",))
        .supply(5, ("chemistry",))
        .build()
    )
    result = resolve_dogma(
        state,
        "self-service",
        choose_card("experimentation"),
        choose_card("gunpowder"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert CardId("chemistry") in result.state.player(P2).board.stack(Color.BLUE).cards
    assert CardId("calendar") in result.state.supply.pile(2)
    assert CardId("calendar") not in result.state.player(P1).score_pile


def test_nested_gunpowder_without_a_demand_is_a_no_op() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("self-service",))
        .board(P1, Color.RED, ("gunpowder",))
        .supply(2, ("calendar",))
        .build()
    )
    result = resolve_dogma(
        state,
        "self-service",
        choose_card("gunpowder"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )

    assert CardId("calendar") in result.state.supply.pile(2)
    assert not result.state.player(P1).score_pile


def test_two_nested_gunpowder_executions_do_not_share_a_demand_result() -> None:
    outer_card = CardId("self-service")
    outer = EffectProgram(
        "double-nested-gunpowder-v1",
        outer_card,
        (ProgramEffect(DogmaEffectId(outer_card, 1), False, "double-nested"),),
        (
            SequenceNode(
                "double-nested",
                ("choose-gunpowder", "execute-first", "execute-second"),
            ),
            ChoiceNode(
                "choose-gunpowder",
                ChoiceKind.CARD,
                "selected-card",
                chooser=EXECUTOR,
                cards=CardSelector.top_cards(EXECUTOR, exclude_source_card=True),
            ),
            NestedNode("execute-first", "selected-card"),
            NestedNode("execute-second", "selected-card"),
        ),
    )
    programs = EffectProgramRegistry(
        (outer, PROGRAMS.program_for_card(CardId("gunpowder"))),
        predicates={
            CardId("gunpowder"): {
                "own-demand-transferred": PROGRAMS.named_predicate(
                    CardId("gunpowder"), "own-demand-transferred"
                )
            }
        },
    )
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("self-service",))
        .board(P1, Color.RED, ("gunpowder",))
        .supply(2, ("calendar",))
        .build()
    )
    context = EffectContext(
        actor=P1,
        chooser=P1,
        executor=P1,
        dogma_activator=P1,
        source_card_id=outer_card,
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
        scope="double-nested",
    )
    paused = start_effect(state, outer.program_id, context, programs, REGISTRY)
    assert paused.status is EffectStatus.AWAIT_DECISION
    decision = paused.decision
    assert decision is not None
    action = next(
        action
        for action in decision.legal_actions
        if isinstance(action, ChooseCardAction) and action.card_id == CardId("gunpowder")
    )
    result = submit_effect_action(paused.state, action, programs, REGISTRY)

    assert result.status is EffectStatus.COMPLETE
    assert CardId("calendar") in result.state.supply.pile(2)
    assert not result.state.player(P1).score_pile
