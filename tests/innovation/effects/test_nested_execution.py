"""Rules decision 6: nested non-demand execution, attribution, and the cycle guard.

Nested execution runs only the selected card's non-demand effects for the current executor. It
starts no share pass, freezes no new icon counts, applies no demands, and awards no separate
sharing bonus - but its changes keep their causal attribution to the outer execution, so an outer
shared execution can still justify exactly one free Draw.
"""

from __future__ import annotations

import pytest

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EXECUTOR,
    MAX_NESTED_DEPTH,
    OPPONENT,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectContext,
    EffectInvariantError,
    EffectProgram,
    EffectProgramRegistry,
    EffectResolution,
    EffectStatus,
    NestedNode,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    UnimplementedCardError,
    ValueRef,
    start_effect,
    submit_effect_action,
)
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    build_explicit_state,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
ROBOTICS = CardId("robotics")
SOFTWARE = CardId("software")


def _nesting_program(program_id: str, card: CardId, target: CardId) -> EffectProgram:
    """A card whose only effect nests unconditionally into ``target``."""

    return EffectProgram(
        program_id,
        card,
        tuple(
            ProgramEffect(DogmaEffectId(card, effect.id.ordinal), False, "nest")
            for effect in REGISTRY.card(card).dogma_effects
        ),
        (
            SequenceNode("nest", ("pick", "execute")),
            ChoiceNode(
                "pick",
                ChoiceKind.CARD,
                "selected",
                cards=CardSelector.constant((target,)),
            ),
            NestedNode("execute", "selected"),
        ),
    )


def _drawing_program(program_id: str, card: CardId) -> EffectProgram:
    return EffectProgram(
        program_id,
        card,
        tuple(
            ProgramEffect(DogmaEffectId(card, effect.id.ordinal), False, "draw")
            for effect in REGISTRY.card(card).dogma_effects
        ),
        (DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),),
    )


def _context(state: GameState, card: CardId, *, shared: bool = False) -> EffectContext:
    return EffectContext(
        actor=P1,
        chooser=P1,
        executor=P1,
        dogma_activator=P1,
        source_card_id=card,
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
        shared=shared,
    )


def _state() -> GameState:
    return build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(board=((Color.RED, (ROBOTICS,)), (Color.BLUE, (SOFTWARE,)))),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
    )


def _resolve(
    state: GameState,
    programs: EffectProgramRegistry,
    program_id: str,
    card: CardId,
    **kwargs: bool,
) -> EffectResolution:
    result = start_effect(state, program_id, _context(state, card, **kwargs), programs, REGISTRY)
    for _ in range(100):
        if result.status is not EffectStatus.AWAIT_DECISION:
            return result
        assert result.decision is not None
        result = submit_effect_action(
            result.state, result.decision.legal_actions[0], programs, REGISTRY
        )
    raise AssertionError("nested resolution did not settle")


def test_nested_execution_runs_the_selected_cards_effects_for_the_same_executor() -> None:
    programs = EffectProgramRegistry(
        (
            _nesting_program("outer-v1", ROBOTICS, SOFTWARE),
            _drawing_program("inner-v1", SOFTWARE),
        )
    )
    state = _state()
    result = _resolve(state, programs, "outer-v1", ROBOTICS)
    assert result.status is EffectStatus.COMPLETE
    # Software prints two effects, so the executor draws twice and the opponent never executes.
    assert len(result.state.player(P1).hand) == 2
    assert not result.state.player(P2).hand


def test_nested_events_are_marked_nested_and_keep_the_outer_source_attribution() -> None:
    programs = EffectProgramRegistry(
        (
            _nesting_program("outer-v1", ROBOTICS, SOFTWARE),
            _drawing_program("inner-v1", SOFTWARE),
        )
    )
    result = _resolve(_state(), programs, "outer-v1", ROBOTICS)
    changes = tuple(event for event in result.events if event.changed)
    assert changes
    for event in changes:
        assert event.nested
        assert not event.demand
        assert event.executor is P1
        assert event.dogma_activator is P1
        # The event names the nested card, so provenance is not lost.
        assert event.source_card_id == SOFTWARE


def test_nested_execution_inherits_the_outer_shared_flag() -> None:
    """Decision 6: outer sharing credit must be retained, not cleared by nesting."""

    programs = EffectProgramRegistry(
        (
            _nesting_program("outer-v1", ROBOTICS, SOFTWARE),
            _drawing_program("inner-v1", SOFTWARE),
        )
    )
    result = _resolve(_state(), programs, "outer-v1", ROBOTICS, shared=True)
    changes = tuple(event for event in result.events if event.changed)
    assert changes
    assert all(event.shared for event in changes)
    assert result.qualifying_changes >= 1


def test_nested_execution_skips_demand_effects() -> None:
    card = SOFTWARE
    printed = REGISTRY.card(card).dogma_effects
    inner = EffectProgram(
        "inner-demand-v1",
        card,
        (
            ProgramEffect(DogmaEffectId(card, printed[0].id.ordinal), False, "draw"),
            ProgramEffect(DogmaEffectId(card, printed[1].id.ordinal), True, "demand-draw"),
        ),
        (
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
            DrawNode("demand-draw", ValueRef.literal(10), "big", player=EXECUTOR),
        ),
    )
    programs = EffectProgramRegistry((_nesting_program("outer-v1", ROBOTICS, SOFTWARE), inner))
    result = _resolve(_state(), programs, "outer-v1", ROBOTICS)
    hand = result.state.player(P1).hand
    # Only the non-demand effect ran, so no age 10 card was drawn.
    assert len(hand) == 1
    assert REGISTRY.card(hand[0]).age == 1


def test_a_nesting_cycle_fails_loudly_at_the_serialized_depth_limit() -> None:
    """Robotics -> Software -> Robotics must raise, not silently truncate."""

    programs = EffectProgramRegistry(
        (
            _nesting_program("outer-v1", ROBOTICS, SOFTWARE),
            _nesting_program("inner-v1", SOFTWARE, ROBOTICS),
        )
    )
    with pytest.raises(EffectInvariantError, match="nested execution depth"):
        _resolve(_state(), programs, "outer-v1", ROBOTICS)


def test_the_nesting_depth_limit_is_an_explicit_shared_constant() -> None:
    assert MAX_NESTED_DEPTH >= 2
    assert isinstance(MAX_NESTED_DEPTH, int)


def test_nesting_into_an_unimplemented_card_fails_loudly() -> None:
    programs = EffectProgramRegistry((_nesting_program("outer-v1", ROBOTICS, CardId("computers")),))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    board=((Color.RED, (ROBOTICS,)), (Color.BLUE, (CardId("computers"),)))
                ),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
    )
    with pytest.raises(UnimplementedCardError):
        _resolve(state, programs, "outer-v1", ROBOTICS)


def test_a_nested_choice_serializes_and_resumes_identically() -> None:
    from innovation_ai.innovation.serialization import dumps_state, loads_state
    from innovation_ai.innovation.state import state_hash

    inner = EffectProgram(
        "inner-choice-v1",
        SOFTWARE,
        tuple(
            ProgramEffect(DogmaEffectId(SOFTWARE, effect.id.ordinal), False, "pick")
            for effect in REGISTRY.card(SOFTWARE).dogma_effects
        ),
        (
            ChoiceNode(
                "pick",
                ChoiceKind.CARD,
                "chosen",
                cards=CardSelector.top_cards(EXECUTOR),
            ),
        ),
    )
    programs = EffectProgramRegistry((_nesting_program("outer-v1", ROBOTICS, SOFTWARE), inner))
    state = _state()
    started = start_effect(state, "outer-v1", _context(state, ROBOTICS), programs, REGISTRY)
    assert started.decision is not None
    live = submit_effect_action(
        started.state, started.decision.legal_actions[0], programs, REGISTRY
    )
    assert live.status is EffectStatus.AWAIT_DECISION
    # This is the nested card's own choice, made by the same executor.
    assert live.decision is not None
    assert live.decision.source is not None
    assert live.decision.source.card_id == SOFTWARE

    restored = loads_state(dumps_state(live.state), REGISTRY)
    assert state_hash(restored) == state_hash(live.state)
    direct = submit_effect_action(live.state, live.decision.legal_actions[0], programs, REGISTRY)
    resumed = submit_effect_action(restored, live.decision.legal_actions[0], programs, REGISTRY)
    assert state_hash(direct.state) == state_hash(resumed.state)


def test_a_nested_rereveal_does_not_cancel_the_outer_reveal() -> None:
    outer = EffectProgram(
        "outer-rereveal-v1",
        ROBOTICS,
        tuple(
            ProgramEffect(DogmaEffectId(ROBOTICS, effect.id.ordinal), False, "outer")
            for effect in REGISTRY.card(ROBOTICS).dogma_effects
        ),
        (
            SequenceNode("outer", ("outer-reveal", "pick-inner", "nested", "pick-hand")),
            RevealNode("outer-reveal", CardSelector.hand(OPPONENT)),
            ChoiceNode(
                "pick-inner",
                ChoiceKind.CARD,
                "selected-inner",
                cards=CardSelector.constant((SOFTWARE,)),
            ),
            NestedNode("nested", "selected-inner"),
            ChoiceNode(
                "pick-hand",
                ChoiceKind.CARD,
                "selected-hand",
                cards=CardSelector.hand(OPPONENT),
            ),
        ),
    )
    inner = EffectProgram(
        "inner-rereveal-v1",
        SOFTWARE,
        tuple(
            ProgramEffect(DogmaEffectId(SOFTWARE, effect.id.ordinal), False, "inner-reveal")
            for effect in REGISTRY.card(SOFTWARE).dogma_effects
        ),
        (RevealNode("inner-reveal", CardSelector.hand(OPPONENT)),),
    )
    programs = EffectProgramRegistry((outer, inner))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(board=((Color.RED, (ROBOTICS,)), (Color.BLUE, (SOFTWARE,)))),
            ),
            (
                P2,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"),),
                    board=((Color.BLUE, (CardId("pottery"),)),),
                ),
            ),
        ),
    )

    started = start_effect(state, outer.program_id, _context(state, ROBOTICS), programs, REGISTRY)
    assert started.decision is not None
    after_nested = submit_effect_action(
        started.state, started.decision.legal_actions[0], programs, REGISTRY
    )
    assert after_nested.status is EffectStatus.AWAIT_DECISION
    assert after_nested.decision is not None
    assert after_nested.decision.source is not None
    assert after_nested.decision.source.card_id == ROBOTICS
    assert CardId("tools") in after_nested.state.revealed_card_ids
    assert {
        action.card_id
        for action in after_nested.decision.legal_actions
        if isinstance(action, ChooseCardAction)
    } == {CardId("tools")}


def test_nesting_with_no_selected_card_is_a_clean_no_op() -> None:
    program = EffectProgram(
        "empty-nest-v1",
        ROBOTICS,
        tuple(
            ProgramEffect(DogmaEffectId(ROBOTICS, effect.id.ordinal), False, "nest")
            for effect in REGISTRY.card(ROBOTICS).dogma_effects
        ),
        (NestedNode("nest", "never-set"),),
    )
    programs = EffectProgramRegistry((program,))
    state = _state()
    result = start_effect(state, "empty-nest-v1", _context(state, ROBOTICS), programs, REGISTRY)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0
