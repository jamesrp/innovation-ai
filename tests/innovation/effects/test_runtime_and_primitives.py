from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import (
    ChooseBranchAction,
    ChooseCardAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseValueAction,
    DrawAction,
)
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EFFECT_RUNTIME_SCHEMA_VERSION,
    EXECUTOR,
    OPPONENT,
    AbortDogmaNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectContext,
    EffectInvariantError,
    EffectProgram,
    EffectProgramRegistry,
    EffectStatus,
    IllegalEffectAction,
    MovementKind,
    MoveNode,
    NoOpNode,
    PlayerRef,
    PlayerRefKind,
    Predicate,
    ProgramEffect,
    RepeatNode,
    SequenceNode,
    SplayNode,
    ValueRef,
    ValueRefKind,
    delete_effect_variable,
    effect_event_payload,
    effect_runtime_payload,
    evaluate_predicate,
    get_effect_variable,
    resolve_player,
    resolve_value,
    restore_effect_runtime,
    resume_effect,
    select_cards,
    set_effect_variable,
    start_effect,
    start_program_effect,
    step_effect,
    submit_effect_action,
)
from innovation_ai.innovation.effects.synthetic import synthetic_program_registry
from innovation_ai.innovation.protocol import IllegalAction
from innovation_ai.innovation.state import (
    EffectFrameState,
    EffectVariable,
    GamePhase,
    GameState,
    build_setup_state,
    state_hash,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    PlayerId,
    SplayDirection,
)
from innovation_ai.innovation.zones import (
    CardLocation,
    ChangeKind,
    ZoneKind,
    locate_card,
    move_card,
    set_splay,
)


def _state() -> GameState:
    return replace(
        build_setup_state(1),
        phase=GamePhase.PLAY,
        active_player=PlayerId.PLAYER_1,
        turn_number=2,
        paid_actions_remaining=1,
    )


def _context(state: GameState, source: CardId, *, step_limit: int = 10_000) -> EffectContext:
    return EffectContext(
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_1,
        source,
        None,
        state.turn_number,
        2,
        step_limit=step_limit,
    )


def _single_node_program(source: CardId, node: object) -> EffectProgram:
    return EffectProgram(
        f"test-{source.value}-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, node.node_id),),  # type: ignore[attr-defined]
        (node,),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("movement", "card_id", "destination_kind"),
    [
        (MovementKind.MELD, CardId("calendar"), ZoneKind.BOARD),
        (MovementKind.TUCK, CardId("calendar"), ZoneKind.BOARD),
        (MovementKind.SCORE, CardId("calendar"), ZoneKind.SCORE),
        (MovementKind.RETURN, CardId("tools"), ZoneKind.SUPPLY),
        (MovementKind.TRANSFER, CardId("tools"), ZoneKind.HAND),
        (MovementKind.REMOVE, CardId("tools"), ZoneKind.REMOVED),
    ],
)
def test_shared_movement_nodes_emit_semantic_provenance(
    movement: MovementKind, card_id: CardId, destination_kind: ZoneKind
) -> None:
    registry = load_card_registry()
    state = _state()
    assert not isinstance(state, type)
    if locate_card(state, card_id).kind is ZoneKind.NORMAL_ACHIEVEMENT:
        pytest.skip("seed unexpectedly set aside a required fixture card")
    if movement is MovementKind.RETURN:
        state, _ = move_card(
            state,
            card_id,
            CardLocation.hand(PlayerId.PLAYER_1),
            registry,
            kind=ChangeKind.TRANSFER,
        )
    source = CardId("calendar") if card_id != CardId("calendar") else CardId("tools")
    if locate_card(state, source).kind is ZoneKind.NORMAL_ACHIEVEMENT:
        pytest.skip("seed unexpectedly set aside a source fixture card")
    kwargs: dict[str, object] = {}
    if movement in {
        MovementKind.MELD,
        MovementKind.TUCK,
        MovementKind.SCORE,
        MovementKind.TRANSFER,
    }:
        kwargs["destination_player"] = OPPONENT if movement is MovementKind.TRANSFER else EXECUTOR
    if movement is MovementKind.TRANSFER:
        kwargs["destination_zone"] = ZoneKind.HAND
    node = MoveNode(
        "move",
        movement,
        CardSelector(CardSelectorKind.CONSTANT, cards=(card_id,)),
        **kwargs,  # type: ignore[arg-type]
    )
    program = _single_node_program(source, node)
    result = start_effect(
        state,
        program.program_id,
        _context(state, source),
        EffectProgramRegistry((program,)),
        registry,
    )
    assert result.status is EffectStatus.COMPLETE
    assert len(result.events) == 1
    event = result.events[0]
    assert event.changed
    assert event.change is not None
    assert event.change.kind.value == movement.value
    assert event.card_ids == (card_id,)
    move = event.change.card_moves[0]
    assert move.destination.kind is destination_kind
    payload = effect_event_payload(event)
    assert payload["executor"] == PlayerId.PLAYER_1.value
    assert payload["source_card_id"] == source.value


def test_all_semantic_choice_shapes_pause_resume_and_reject_illegal_actions() -> None:
    state = _state()
    source = CardId("calendar")
    nodes = (
        SequenceNode("all", ("player", "value", "direction", "branch")),
        ChoiceNode(
            "player",
            ChoiceKind.PLAYER,
            "player-choice",
            players=(EXECUTOR, OPPONENT),
        ),
        ChoiceNode(
            "value",
            ChoiceKind.VALUE,
            "value-choice",
            values=(1, 5, 10),
        ),
        ChoiceNode(
            "direction",
            ChoiceKind.SPLAY,
            "direction-choice",
            directions=(SplayDirection.LEFT, SplayDirection.RIGHT, SplayDirection.UP),
        ),
        ChoiceNode(
            "branch",
            ChoiceKind.BRANCH,
            "branch-choice",
            branches=("alpha", "omega"),
        ),
    )
    program = EffectProgram(
        "test-choice-shapes-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "all"),),
        nodes,
    )
    programs = EffectProgramRegistry((program,))
    result = start_effect(state, program.program_id, _context(state, source), programs)
    assert result.status is EffectStatus.AWAIT_DECISION
    assert all(isinstance(action, ChoosePlayerAction) for action in result.decision.legal_actions)  # type: ignore[union-attr]
    with pytest.raises(IllegalEffectAction):
        submit_effect_action(result.state, DrawAction(result.state.next_decision_id), programs)

    expected_types = (
        ChoosePlayerAction,
        ChooseValueAction,
        ChooseSplayAction,
        ChooseBranchAction,
    )
    for expected in expected_types:
        assert result.decision is not None
        assert all(isinstance(action, expected) for action in result.decision.legal_actions)
        result = submit_effect_action(
            result.state,
            result.decision.legal_actions[-1],
            programs,
        )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.next_decision_id == state.next_decision_id + 4


def test_serialized_step_ceiling_fails_loudly_and_runtime_schema_is_versioned() -> None:
    state = _state()
    source = CardId("calendar")
    program = EffectProgram(
        "test-step-limit-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "steps"),),
        (
            SequenceNode("steps", ("one", "two", "three")),
            NoOpNode("one"),
            NoOpNode("two"),
            NoOpNode("three"),
        ),
    )
    programs = EffectProgramRegistry((program,))
    context = _context(state, source, step_limit=2)
    started = start_effect(
        state,
        program.program_id,
        context,
        programs,
        pause_before_first_step=True,
    )
    payload = effect_runtime_payload(started.state)
    assert payload["schema_version"] == EFFECT_RUNTIME_SCHEMA_VERSION
    restored = restore_effect_runtime(
        replace(started.state, pending_effects=(), effect_variables=()), payload
    )
    with pytest.raises(EffectInvariantError, match="step limit"):
        resume_effect(restored, programs)

    bad_payload = dict(payload)
    bad_payload["schema_version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        restore_effect_runtime(started.state, bad_payload)


def test_effect_context_and_program_validation_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="effect scope"):
        EffectContext(
            PlayerId.PLAYER_1,
            PlayerId.PLAYER_1,
            PlayerId.PLAYER_1,
            PlayerId.PLAYER_1,
            CardId("calendar"),
            None,
            1,
            1,
            scope="not valid",
        )
    with pytest.raises(ValueError, match="literal player"):
        PlayerRef(PlayerRefKind.LITERAL)


def test_reentered_bounded_choice_resets_scope_and_mandatory_partial_finishes() -> None:
    state = _state()
    source = CardId("calendar")
    candidate = CardId("tools")
    repeated = EffectProgram(
        "test-repeat-choice-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "repeat"),),
        (
            RepeatNode(
                "repeat",
                "choose",
                Predicate.truthy("selected"),
                maximum_iterations=2,
            ),
            ChoiceNode(
                "choose",
                ChoiceKind.BOUNDED_CARDS,
                "selected",
                cards=CardSelector(CardSelectorKind.CONSTANT, cards=(candidate,)),
                minimum=0,
                maximum=2,
            ),
        ),
    )
    programs = EffectProgramRegistry((repeated,))
    result = start_effect(state, repeated.program_id, _context(state, source), programs)
    assert result.decision is not None
    choose = next(
        action
        for action in result.decision.legal_actions
        if getattr(action, "card_id", None) == candidate
    )
    result = submit_effect_action(result.state, choose, programs)
    assert result.status is EffectStatus.AWAIT_DECISION
    assert result.decision is not None
    assert any(
        getattr(action, "card_id", None) == candidate for action in result.decision.legal_actions
    )

    partial = EffectProgram(
        "test-partial-choice-v1",
        CardId("pottery"),
        (ProgramEffect(DogmaEffectId(CardId("pottery"), 1), False, "partial"),),
        (
            ChoiceNode(
                "partial",
                ChoiceKind.BOUNDED_CARDS,
                "selected",
                cards=CardSelector(CardSelectorKind.CONSTANT, cards=(candidate,)),
                minimum=2,
                maximum=2,
            ),
        ),
    )
    partial_programs = EffectProgramRegistry((partial,))
    result = start_effect(
        state,
        partial.program_id,
        _context(state, CardId("pottery")),
        partial_programs,
    )
    assert result.decision is not None
    result = submit_effect_action(result.state, result.decision.legal_actions[0], partial_programs)
    assert result.status is EffectStatus.COMPLETE


def test_canonical_bounded_selection_never_strands_a_reachable_minimum() -> None:
    state = _state()
    source = CardId("pottery")
    lower = CardId("tools")
    higher = CardId("writing")
    program = EffectProgram(
        "test-bounded-minimum-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "choose"),),
        (
            ChoiceNode(
                "choose",
                ChoiceKind.BOUNDED_CARDS,
                "selected",
                cards=CardSelector.constant((lower, higher)),
                minimum=2,
                maximum=2,
            ),
        ),
    )
    programs = EffectProgramRegistry((program,))

    started = start_effect(state, program.program_id, _context(state, source), programs)
    assert started.decision is not None
    assert started.decision.legal_actions == (
        ChooseCardAction(started.decision.decision_id, lower),
    )
    continued = submit_effect_action(started.state, started.decision.legal_actions[0], programs)
    assert continued.decision is not None
    assert continued.decision.legal_actions == (
        ChooseCardAction(continued.decision.decision_id, higher),
    )
    completed = submit_effect_action(continued.state, continued.decision.legal_actions[0], programs)
    assert completed.status is EffectStatus.COMPLETE


def test_effect_scopes_do_not_bleed_and_dynamic_choices_are_consumable() -> None:
    registry = load_card_registry()
    state = _state()
    source = CardId("calendar")
    state, _ = move_card(
        state,
        CardId("tools"),
        CardLocation.hand(PlayerId.PLAYER_1),
        registry,
        kind=ChangeKind.TRANSFER,
    )
    state, _ = move_card(
        state,
        CardId("pottery"),
        CardLocation.board(PlayerId.PLAYER_1, Color.BLUE),
        registry,
        kind=ChangeKind.MELD,
    )
    state, _ = move_card(
        state,
        CardId("calendar"),
        CardLocation.board(PlayerId.PLAYER_1, Color.BLUE),
        registry,
        kind=ChangeKind.MELD,
    )
    state, _ = set_splay(state, PlayerId.PLAYER_1, Color.BLUE, SplayDirection.RIGHT, registry)
    program = EffectProgram(
        "test-scopes-and-dynamic-v1",
        source,
        (
            ProgramEffect(DogmaEffectId(source, 1), False, "first"),
            ProgramEffect(DogmaEffectId(source, 2), False, "second"),
        ),
        (
            ChoiceNode(
                "first",
                ChoiceKind.BRANCH,
                "selection",
                branches=("selected",),
            ),
            SequenceNode(
                "second",
                ("choose-player", "choose-color", "choose-direction", "if-selection"),
            ),
            ChoiceNode(
                "choose-player",
                ChoiceKind.PLAYER,
                "target",
                players=(EXECUTOR, OPPONENT),
            ),
            ChoiceNode(
                "choose-color",
                ChoiceKind.COLOR,
                "color",
                colors=(Color.BLUE,),
            ),
            ChoiceNode(
                "choose-direction",
                ChoiceKind.SPLAY,
                "direction",
                directions=(SplayDirection.LEFT, SplayDirection.UP),
            ),
            ConditionNode(
                "if-selection",
                Predicate.equals("selection", "selected"),
                "should-not-run",
                "dynamic-splay",
            ),
            MoveNode(
                "should-not-run",
                MovementKind.TRANSFER,
                CardSelector(CardSelectorKind.CONSTANT, cards=(CardId("tools"),)),
                destination_player=PlayerRef.from_variable("target"),
                destination_zone=ZoneKind.HAND,
            ),
            SplayNode(
                "dynamic-splay",
                EXECUTOR,
                color_variable="color",
                direction_variable="direction",
            ),
        ),
    )
    programs = EffectProgramRegistry((program,))
    result = start_effect(state, program.program_id, _context(state, source), programs)
    for expected_kind in ("choose-branch", "choose-player", "choose-color", "choose-splay"):
        assert result.decision is not None
        action = result.decision.legal_actions[-1]
        assert action.kind.value == expected_kind
        result = submit_effect_action(result.state, action, programs, registry)
    assert result.status is EffectStatus.COMPLETE
    assert CardId("tools") in result.state.player(PlayerId.PLAYER_1).hand
    assert result.state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE).splay is SplayDirection.UP


def test_movement_result_variable_supports_if_you_do_conditions() -> None:
    registry = load_card_registry()
    state = _state()
    source = CardId("pottery")
    program = EffectProgram(
        "test-if-you-do-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "all"),),
        (
            SequenceNode("all", ("transfer", "if-transfer")),
            MoveNode(
                "transfer",
                MovementKind.TRANSFER,
                CardSelector(CardSelectorKind.CONSTANT, cards=(CardId("tools"),)),
                destination_player=OPPONENT,
                destination_zone=ZoneKind.HAND,
                result_variable="did-transfer",
            ),
            ConditionNode(
                "if-transfer",
                Predicate.truthy("did-transfer"),
                "score-reward",
            ),
            MoveNode(
                "score-reward",
                MovementKind.SCORE,
                CardSelector(CardSelectorKind.CONSTANT, cards=(CardId("calendar"),)),
                destination_player=EXECUTOR,
            ),
        ),
    )
    result = start_effect(
        state,
        program.program_id,
        _context(state, source),
        EffectProgramRegistry((program,)),
        registry,
    )
    assert result.status is EffectStatus.COMPLETE
    assert CardId("tools") in result.state.player(PlayerId.PLAYER_2).hand
    assert CardId("calendar") in result.state.player(PlayerId.PLAYER_1).score_pile
    assert result.qualifying_changes == 2


def test_start_single_program_effect_supports_wp5_executor_ordering() -> None:
    registry = load_card_registry()
    state = _state()
    state, _ = move_card(
        state,
        CardId("machinery"),
        CardLocation.board(PlayerId.PLAYER_1, Color.YELLOW),
        registry,
        kind=ChangeKind.MELD,
    )
    programs = synthetic_program_registry()
    result = start_program_effect(
        state,
        "synthetic-machinery-v1",
        1,
        replace(
            _context(state, CardId("machinery")),
            actor=PlayerId.PLAYER_2,
            chooser=PlayerId.PLAYER_2,
            executor=PlayerId.PLAYER_2,
        ),
        programs,
        registry,
    )
    assert result.status is EffectStatus.COMPLETE
    assert all(event.demand for event in result.events)
    with pytest.raises(EffectInvariantError, match="no effect ordinal"):
        start_program_effect(
            state,
            "synthetic-machinery-v1",
            3,
            replace(
                _context(state, CardId("machinery")),
                actor=PlayerId.PLAYER_2,
                chooser=PlayerId.PLAYER_2,
                executor=PlayerId.PLAYER_2,
            ),
            programs,
            registry,
        )


def test_root_effect_rejects_preexisting_runtime_instead_of_nesting_implicitly() -> None:
    """Nested execution must use ``NestedNode`` rather than stacking another root runtime."""

    state = _state()
    base = EffectFrameState(
        "external-orchestrator",
        variables=(EffectVariable("owner", "integration"),),
    )
    state = replace(
        state,
        pending_effects=(base,),
        effect_variables=(EffectVariable("orchestrator:index", 2),),
    )
    source = CardId("calendar")
    programs = synthetic_program_registry()
    context = replace(_context(state, source), scope="executor-1")
    before = state_hash(state)
    with pytest.raises(EffectInvariantError, match="runtime is pending"):
        start_effect(
            state,
            "synthetic-bounded-selection-v1",
            context,
            programs,
        )
    assert state_hash(state) == before


def test_step_effect_abort_reports_persisted_change_count() -> None:
    registry = load_card_registry()
    state = _state()
    source = CardId("pottery")
    program = EffectProgram(
        "test-abort-count-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "all"),),
        (
            SequenceNode("all", ("move", "abort")),
            MoveNode(
                "move",
                MovementKind.TRANSFER,
                CardSelector(CardSelectorKind.CONSTANT, cards=(CardId("tools"),)),
                destination_player=OPPONENT,
                destination_zone=ZoneKind.HAND,
            ),
            AbortDogmaNode("abort"),
        ),
    )
    programs = EffectProgramRegistry((program,))
    result = start_effect(
        state,
        program.program_id,
        _context(state, source),
        programs,
        registry,
        pause_before_first_step=True,
    )
    while result.status is EffectStatus.CONTINUE:
        result = step_effect(result.state, programs, registry)
    assert result.status is EffectStatus.ABORT_DOGMA
    assert result.qualifying_changes == 1


def test_absent_card_predicate_is_false_and_contract_validation_is_strict() -> None:
    state = _state()
    source = CardId("calendar")
    program = EffectProgram(
        "test-absent-predicate-v1",
        source,
        (ProgramEffect(DogmaEffectId(source, 1), False, "condition"),),
        (
            ConditionNode(
                "condition",
                Predicate.card_color_is("missing", Color.RED),
                "bad",
                "good",
            ),
            AbortDogmaNode("bad"),
            NoOpNode("good"),
        ),
    )
    result = start_effect(
        state,
        program.program_id,
        _context(state, source),
        EffectProgramRegistry((program,)),
    )
    assert result.status is EffectStatus.COMPLETE

    with pytest.raises(ValueError, match="positive maximum"):
        ChoiceNode(
            "invalid",
            ChoiceKind.BOUNDED_CARDS,
            "cards",
            cards=CardSelector.hand(),
            minimum=0,
            maximum=0,
        )
    with pytest.raises(ValueError, match="unique and increasing"):
        EffectProgram(
            "duplicate-ordinal-v1",
            source,
            (
                ProgramEffect(DogmaEffectId(source, 1), False, "good"),
                ProgramEffect(DogmaEffectId(source, 1), False, "good"),
            ),
            (NoOpNode("good"),),
        )


def test_restore_rejects_missing_vm_counters_and_effect_illegal_is_wp3_illegal() -> None:
    state = _state()
    source = CardId("calendar")
    programs = synthetic_program_registry()
    started = start_effect(
        state,
        "synthetic-bounded-selection-v1",
        _context(state, source),
        programs,
    )
    payload = effect_runtime_payload(started.state)
    variables = payload["effect_variables"]
    assert isinstance(variables, list)
    payload["effect_variables"] = [
        item
        for item in variables
        if isinstance(item, dict) and item.get("name") != "root:step-count"
    ]
    with pytest.raises(ValueError, match="missing serialized VM counters"):
        restore_effect_runtime(
            replace(started.state, pending_effects=(), effect_variables=()),
            payload,
            programs,
        )

    assert started.decision is not None
    with pytest.raises(IllegalEffectAction) as raised:
        submit_effect_action(
            started.state,
            DrawAction(started.decision.decision_id),
            programs,
        )
    assert isinstance(raised.value, IllegalAction)


def test_reference_selector_predicate_and_scope_helpers_cover_declarative_surface() -> None:
    registry = load_card_registry()
    state = _state()
    context = _context(state, CardId("calendar"))
    state = set_effect_variable(state, context, "target", PlayerId.PLAYER_2.value)
    state = set_effect_variable(state, context, "number", 5)
    state = set_effect_variable(
        state,
        context,
        "cards",
        (CardId("tools").value, CardId("pottery").value),
    )
    references = (
        (PlayerRef(PlayerRefKind.ACTOR), context.actor),
        (PlayerRef(PlayerRefKind.CHOOSER), context.chooser),
        (PlayerRef(PlayerRefKind.EXECUTOR), context.executor),
        (PlayerRef(PlayerRefKind.ACTIVATOR), context.dogma_activator),
        (PlayerRef(PlayerRefKind.OPPONENT_OF_EXECUTOR), PlayerId.PLAYER_2),
        (PlayerRef.literal(PlayerId.PLAYER_2), PlayerId.PLAYER_2),
        (PlayerRef.from_variable("target"), PlayerId.PLAYER_2),
    )
    for reference, expected in references:
        assert resolve_player(reference, context, state) is expected

    assert resolve_value(state, context, ValueRef.literal(3)) == 3
    assert resolve_value(state, context, ValueRef(ValueRefKind.VARIABLE, variable="number")) == 5
    assert resolve_value(state, context, ValueRef.count("cards")) == 2
    assert select_cards(state, context, CardSelector.from_variable("cards"), registry) == (
        CardId("tools"),
        CardId("pottery"),
    )
    assert select_cards(state, context, CardSelector.hand(), registry) == state.players[0].hand
    assert evaluate_predicate(state, context, Predicate.truthy("cards"), registry)
    assert evaluate_predicate(state, context, Predicate.equals("number", 5), registry)
    state = set_effect_variable(state, context, "one-card", CardId("tools").value)
    assert evaluate_predicate(
        state,
        context,
        Predicate.card_color_is("one-card", registry.card("tools").color),
        registry,
    )
    icon = registry.card("tools").functional_icons[0]
    assert evaluate_predicate(state, context, Predicate.card_has_icon("one-card", icon), registry)
    assert get_effect_variable(state, context, "number") == 5
    state = delete_effect_variable(state, context, "number")
    assert get_effect_variable(state, context, "number") is None
