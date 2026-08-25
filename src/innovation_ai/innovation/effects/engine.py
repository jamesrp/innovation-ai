"""Explicit stack interpreter for resumable Innovation effects."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import cast

from innovation_ai.innovation.actions import (
    ChooseBranchAction,
    ChooseCardAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseValueAction,
    Decision,
    DecisionKind,
    DecisionSource,
    DeclineAction,
    FinishSelectionAction,
    OrderCardsAction,
    SemanticAction,
)
from innovation_ai.innovation.board import score_value, top_cards
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.state import (
    EffectFrameState,
    GamePhase,
    GameState,
    TerminalReason,
    TerminalState,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import (
    CardLocation,
    ChangeKind,
    ChangeRecord,
    SplayChange,
    ZoneKind,
    draw_card,
    exchange_cards,
    meld_card,
    move_card,
    rearrange_stack,
    remove_card,
    return_card,
    score_card,
    set_splay,
    tuck_card,
)

from .model import (
    EffectContext,
    EffectEvent,
    EffectEventKind,
    EffectInvariantError,
    EffectResolution,
    EffectStatus,
    IllegalEffectAction,
    clear_effect_scope,
    delete_effect_variable,
    frame_context,
    frame_value,
    get_effect_variable,
    make_frame,
    set_effect_variable,
)
from .program import (
    AbortDogmaNode,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectNode,
    EffectProgram,
    EffectProgramRegistry,
    ExchangeNode,
    KeepNode,
    MovementKind,
    MoveNode,
    NestedNode,
    NoOpNode,
    PlayerRef,
    PlayerRefKind,
    Predicate,
    PredicateKind,
    ProgramEffect,
    RearrangeNode,
    RemoveAllPlayCardsNode,
    RepeatNode,
    RevealNode,
    SequenceNode,
    SplayNode,
    ValueRef,
    ValueRefKind,
)

_PROGRAM_FRAME = "effect-program"
_NODE_FRAME = "effect-node"
_LAST_CHOOSER = "last-chooser"
_STEP_COUNT = "step-count"
_QUALIFYING_CHANGE_COUNT = "qualifying-change-count"
_NESTED_COUNT = "nested-count"


def other_player(player_id: PlayerId) -> PlayerId:
    """Return the opponent in the supported two-player game."""

    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def resolve_player(
    reference: PlayerRef,
    context: EffectContext,
    state: GameState | None = None,
) -> PlayerId:
    """Resolve a declarative player reference against live effect context."""

    if reference.kind is PlayerRefKind.ACTOR:
        return context.actor
    if reference.kind is PlayerRefKind.CHOOSER:
        return context.chooser
    if reference.kind is PlayerRefKind.EXECUTOR:
        return context.executor
    if reference.kind is PlayerRefKind.ACTIVATOR:
        return context.dogma_activator
    if reference.kind is PlayerRefKind.OPPONENT_OF_EXECUTOR:
        return other_player(context.executor)
    if reference.kind is PlayerRefKind.VARIABLE:
        if state is None or reference.variable is None:
            raise EffectInvariantError("variable player reference requires live effect state")
        value = get_effect_variable(state, context, reference.variable)
        if not isinstance(value, str):
            raise EffectInvariantError(f"effect variable {reference.variable!r} is not a player ID")
        return PlayerId(value)
    if reference.player_id is None:  # pragma: no cover - validated by PlayerRef
        raise EffectInvariantError("literal player reference is missing its player ID")
    return reference.player_id


def _text_variable(state: GameState, context: EffectContext, name: str) -> str | None:
    value = get_effect_variable(state, context, name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EffectInvariantError(f"effect variable {name!r} is not text")
    return value


def _card_variable(state: GameState, context: EffectContext, name: str) -> CardId | None:
    value = _text_variable(state, context, name)
    return CardId(value) if value is not None else None


def _card_tuple_variable(state: GameState, context: EffectContext, name: str) -> tuple[CardId, ...]:
    value = get_effect_variable(state, context, name, ())
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise EffectInvariantError(f"effect variable {name!r} is not a card tuple")
    return tuple(CardId(item) for item in cast(tuple[str, ...], value))


def _color_variable(state: GameState, context: EffectContext, name: str) -> Color | None:
    value = _text_variable(state, context, name)
    return Color(value) if value is not None else None


def _resolve_color(state: GameState, context: EffectContext, selector: CardSelector) -> Color:
    if selector.color is not None:
        return selector.color
    if selector.color_variable is None:  # pragma: no cover - validated by selector
        raise EffectInvariantError("board-stack selector has no color source")
    color = _color_variable(state, context, selector.color_variable)
    if color is None:
        raise EffectInvariantError("board-stack color variable has no selection")
    return color


def select_cards(
    state: GameState,
    context: EffectContext,
    selector: CardSelector,
    registry: CardRegistry,
) -> tuple[CardId, ...]:
    """Evaluate a deterministic card selector without exposing callbacks."""

    if selector.kind is CardSelectorKind.VARIABLE:
        assert selector.variable is not None
        value = get_effect_variable(state, context, selector.variable)
        if value is None:
            cards: tuple[CardId, ...] = ()
        elif isinstance(value, str):
            cards = (CardId(value),)
        elif isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            cards = tuple(CardId(item) for item in cast(tuple[str, ...], value))
        else:
            raise EffectInvariantError(f"variable {selector.variable!r} is not a card selection")
    elif selector.kind is CardSelectorKind.CONSTANT:
        cards = selector.cards
    else:
        if selector.player is None:  # pragma: no cover - validated by selector
            raise EffectInvariantError("zone selector is missing a player")
        player_id = resolve_player(selector.player, context, state)
        player = state.player(player_id)
        if selector.kind is CardSelectorKind.HAND:
            cards = player.hand
        elif selector.kind is CardSelectorKind.SCORE:
            cards = player.score_pile
        elif selector.kind is CardSelectorKind.BOARD_STACK:
            cards = player.board.stack(_resolve_color(state, context, selector)).cards
        elif selector.kind is CardSelectorKind.TOP_CARDS:
            cards = top_cards(player.board)
        else:  # pragma: no cover - exhaustive guard
            raise EffectInvariantError(f"unhandled card selector: {selector.kind}")

    if selector.icon is not None:
        cards = tuple(
            card_id for card_id in cards if selector.icon in registry.card(card_id).functional_icons
        )
    if selector.highest_only and cards:
        highest = max(registry.card(card_id).age for card_id in cards)
        cards = tuple(card_id for card_id in cards if registry.card(card_id).age == highest)
    if selector.exclude_source_card:
        cards = tuple(card_id for card_id in cards if card_id != context.source_card_id)
    return cards


def _selector_location(
    state: GameState, context: EffectContext, selector: CardSelector
) -> CardLocation:
    if selector.player is None:
        raise EffectInvariantError("exchange selectors must address concrete player zones")
    player_id = resolve_player(selector.player, context, state)
    if selector.kind is CardSelectorKind.HAND:
        return CardLocation.hand(player_id)
    if selector.kind is CardSelectorKind.SCORE:
        return CardLocation.score(player_id)
    if selector.kind is CardSelectorKind.BOARD_STACK:
        return CardLocation.board(player_id, _resolve_color(state, context, selector))
    raise EffectInvariantError("exchange selectors must address a hand, score, or board stack")


def resolve_value(state: GameState, context: EffectContext, reference: ValueRef) -> int:
    """Evaluate a small integer expression from scoped serializable values."""

    if reference.kind is ValueRefKind.LITERAL:
        assert reference.value is not None
        return reference.value
    assert reference.variable is not None
    value = get_effect_variable(state, context, reference.variable)
    if reference.kind is ValueRefKind.COUNT_CARDS:
        if value is None:
            return 0
        if isinstance(value, str):
            return 1
        if isinstance(value, tuple):
            return len(value)
        raise EffectInvariantError(f"variable {reference.variable!r} is not countable")
    if not isinstance(value, int) or isinstance(value, bool):
        raise EffectInvariantError(f"variable {reference.variable!r} is not an integer")
    return value


def evaluate_predicate(
    state: GameState,
    context: EffectContext,
    predicate: Predicate,
    registry: CardRegistry,
) -> bool:
    """Evaluate an explicit condition against current state and scoped variables."""

    value = get_effect_variable(state, context, predicate.variable)
    if predicate.kind is PredicateKind.VARIABLE_TRUTHY:
        return bool(value)
    if predicate.kind is PredicateKind.VARIABLE_EQUALS:
        return value == predicate.expected
    if value is None:
        return False
    if not isinstance(value, str):
        raise EffectInvariantError(
            f"card predicate variable {predicate.variable!r} is not a card ID"
        )
    card = registry.card(CardId(value))
    if predicate.kind is PredicateKind.CARD_HAS_ICON:
        assert predicate.icon is not None
        return predicate.icon in card.functional_icons
    assert predicate.color is not None
    return card.color is predicate.color


def _program_id(frame: EffectFrameState) -> str:
    value = frame_value(frame, "program_id")
    if not isinstance(value, str):
        raise EffectInvariantError("effect frame has no program ID")
    return value


def _node_id(frame: EffectFrameState) -> str:
    value = frame_value(frame, "node_id")
    if not isinstance(value, str):
        raise EffectInvariantError("effect frame has no node ID")
    return value


def _non_demand_only(frame: EffectFrameState) -> bool:
    value = frame_value(frame, "non_demand_only", False)
    if not isinstance(value, bool):
        raise EffectInvariantError("program frame has an invalid nested mode")
    return value


def _is_root_program(frame: EffectFrameState) -> bool:
    value = frame_value(frame, "root_program", False)
    if not isinstance(value, bool):
        raise EffectInvariantError("program frame has an invalid root marker")
    return value


def _selected_effect_ordinal(frame: EffectFrameState) -> int:
    value = frame_value(frame, "selected_effect_ordinal", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EffectInvariantError("program frame has an invalid selected effect ordinal")
    return value


def _replace_top(state: GameState, frame: EffectFrameState) -> GameState:
    if not state.pending_effects:
        raise EffectInvariantError("cannot replace an empty effect stack")
    return replace(state, pending_effects=(*state.pending_effects[:-1], frame))


def _pop(state: GameState) -> GameState:
    if not state.pending_effects:
        raise EffectInvariantError("cannot pop an empty effect stack")
    return replace(state, pending_effects=state.pending_effects[:-1])


def _push(state: GameState, frame: EffectFrameState) -> GameState:
    return replace(state, pending_effects=(*state.pending_effects, frame))


def _node_frame(program_id: str, node_id: str, context: EffectContext) -> EffectFrameState:
    return make_frame(_NODE_FRAME, context, program_id=program_id, node_id=node_id)


def _program_frame(
    program: EffectProgram,
    context: EffectContext,
    *,
    non_demand_only: bool,
    selected_effect_ordinal: int = 0,
    root_program: bool = False,
) -> EffectFrameState:
    return make_frame(
        _PROGRAM_FRAME,
        context,
        program_id=program.program_id,
        non_demand_only=non_demand_only,
        selected_effect_ordinal=selected_effect_ordinal,
        root_program=root_program,
    )


def _root_context(context: EffectContext) -> EffectContext:
    return replace(context, scope=context.scope.split("/", maxsplit=1)[0])


def _bump_step(state: GameState, context: EffectContext) -> GameState:
    root = _root_context(context)
    value = get_effect_variable(state, root, _STEP_COUNT, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EffectInvariantError("serialized effect step count is invalid")
    if value >= context.step_limit:
        raise EffectInvariantError(
            f"effect step limit {context.step_limit} exceeded in dogma action "
            f"{context.dogma_action_id}"
        )
    return set_effect_variable(state, root, _STEP_COUNT, value + 1)


def _effective_context(state: GameState, context: EffectContext) -> EffectContext:
    chooser = get_effect_variable(state, context, _LAST_CHOOSER)
    if chooser is None:
        return context
    if not isinstance(chooser, str):
        raise EffectInvariantError("serialized last chooser is invalid")
    return replace(context, chooser=PlayerId(chooser))


def _event(
    state: GameState,
    context: EffectContext,
    kind: EffectEventKind,
    *,
    change: ChangeRecord | None = None,
    card_ids: tuple[CardId, ...] = (),
    atomic_group_id: int | None = None,
) -> tuple[GameState, EffectEvent]:
    causal = _effective_context(state, context)
    qualifying = kind is EffectEventKind.REVEAL or (
        kind is EffectEventKind.CHANGE and change is not None and change.changed
    )
    if qualifying:
        root = _root_context(context)
        count = get_effect_variable(state, root, _QUALIFYING_CHANGE_COUNT, 0)
        if not isinstance(count, int) or isinstance(count, bool):
            raise EffectInvariantError("serialized qualifying-change count is invalid")
        state = set_effect_variable(state, root, _QUALIFYING_CHANGE_COUNT, count + 1)
    group_id = atomic_group_id if atomic_group_id is not None else state.next_event_id
    event = EffectEvent(
        state.next_event_id,
        kind,
        causal.actor,
        causal.chooser,
        causal.executor,
        causal.dogma_activator,
        causal.source_card_id,
        causal.source_effect_id,
        causal.turn_id,
        causal.dogma_action_id,
        causal.demand,
        causal.shared,
        causal.nested,
        change,
        card_ids,
        group_id,
    )
    return replace(state, next_event_id=state.next_event_id + 1), event


def qualifying_change_count(state: GameState, context: EffectContext) -> int:
    """Return the persisted number of sharing-qualifying changes for this root execution."""

    value = get_effect_variable(state, _root_context(context), _QUALIFYING_CHANGE_COUNT, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EffectInvariantError("serialized qualifying-change count is invalid")
    return value


def _combine_changes(kind: ChangeKind, changes: tuple[ChangeRecord, ...]) -> ChangeRecord:
    moves = tuple(move for change in changes for move in change.card_moves)
    splays: list[SplayChange] = []
    seen: set[tuple[PlayerId, Color]] = set()
    for change in changes:
        for splay in change.splay_changes:
            key = (splay.player_id, splay.color)
            if key in seen:
                continue
            seen.add(key)
            splays.append(splay)
    return ChangeRecord(kind, moves, tuple(splays))


def _draw_exhaustion_result(state: GameState, registry: CardRegistry) -> TerminalState:
    scores = {player_id: score_value(state.player(player_id), registry) for player_id in PlayerId}
    highest = max(scores.values())
    candidates = tuple(player_id for player_id in PlayerId if scores[player_id] == highest)
    if len(candidates) == 1:
        return TerminalState(TerminalReason.DRAW_BEYOND_AGE_10, candidates)
    achievements = {
        player_id: state.player(player_id).achievement_count for player_id in candidates
    }
    most = max(achievements.values())
    winners = tuple(player_id for player_id in candidates if achievements[player_id] == most)
    return TerminalState(TerminalReason.DRAW_BEYOND_AGE_10, winners if len(winners) == 1 else ())


def _movement_change(
    state: GameState,
    context: EffectContext,
    node: MoveNode,
    registry: CardRegistry,
) -> tuple[GameState, ChangeRecord]:
    cards = select_cards(state, context, node.cards, registry)
    changes: list[ChangeRecord] = []
    updated = state
    for card_id in cards:
        if node.movement is MovementKind.RETURN:
            updated, change = return_card(updated, card_id, registry)
        elif node.movement is MovementKind.REMOVE:
            updated, change = remove_card(updated, card_id, registry)
        elif node.movement is MovementKind.MELD:
            assert node.destination_player is not None
            updated, change = meld_card(
                updated,
                resolve_player(node.destination_player, context, updated),
                card_id,
                registry,
            )
        elif node.movement is MovementKind.TUCK:
            assert node.destination_player is not None
            updated, change = tuck_card(
                updated,
                resolve_player(node.destination_player, context, updated),
                card_id,
                registry,
            )
        elif node.movement is MovementKind.SCORE:
            assert node.destination_player is not None
            updated, change = score_card(
                updated,
                resolve_player(node.destination_player, context, updated),
                card_id,
                registry,
            )
        else:
            assert node.destination_player is not None and node.destination_zone is not None
            player = resolve_player(node.destination_player, context, updated)
            destination = (
                CardLocation.hand(player)
                if node.destination_zone is ZoneKind.HAND
                else CardLocation.score(player)
            )
            updated, change = move_card(
                updated,
                card_id,
                destination,
                registry,
                kind=ChangeKind.TRANSFER,
            )
        changes.append(change)
    return updated, _combine_changes(ChangeKind(node.movement.value), tuple(changes))


def _remove_all_play_cards(
    state: GameState, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    ordered: list[CardId] = []
    for player_id in PlayerId:
        player = state.player(player_id)
        ordered.extend(player.hand)
        for color in Color:
            ordered.extend(player.board.stack(color).cards)
        ordered.extend(player.score_pile)
    changes: list[ChangeRecord] = []
    updated = state
    for card_id in ordered:
        updated, change = remove_card(updated, card_id, registry)
        changes.append(change)
    return updated, _combine_changes(ChangeKind.REMOVE, tuple(changes))


def _execute_leaf(
    state: GameState,
    context: EffectContext,
    node: EffectNode,
    registry: CardRegistry,
    *,
    atomic_group_id: int | None = None,
) -> tuple[GameState, tuple[EffectEvent, ...], EffectStatus]:
    events: list[EffectEvent] = []
    updated = state
    if isinstance(node, DrawNode):
        player = resolve_player(node.player, context, updated)
        updated, change, result = draw_card(
            updated, resolve_value(updated, context, node.requested_age), player, registry
        )
        value = result.card_id.value if result.card_id is not None else None
        updated = set_effect_variable(updated, context, node.result_variable, value)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                card_ids=(result.card_id,) if result.card_id is not None else (),
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
        if result.beyond_age_ten:
            terminal = _draw_exhaustion_result(updated, registry)
            updated = replace(
                updated,
                phase=GamePhase.TERMINAL,
                terminal_result=terminal,
                pending_effects=(),
            )
            return updated, tuple(events), EffectStatus.TERMINAL
    elif isinstance(node, RevealNode | KeepNode):
        cards = select_cards(updated, context, node.cards, registry)
        if cards:
            kind = EffectEventKind.REVEAL if isinstance(node, RevealNode) else EffectEventKind.KEEP
            updated, event = _event(
                updated,
                context,
                kind,
                card_ids=cards,
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, MoveNode):
        updated, change = _movement_change(updated, context, node, registry)
        if node.result_variable is not None:
            updated = set_effect_variable(updated, context, node.result_variable, change.changed)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                card_ids=tuple(move.card_id for move in change.card_moves),
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, ExchangeNode):
        first_location = _selector_location(updated, context, node.first)
        second_location = _selector_location(updated, context, node.second)
        first_cards = select_cards(updated, context, node.first, registry)
        second_cards = select_cards(updated, context, node.second, registry)
        updated, change = exchange_cards(
            updated,
            first_location,
            first_cards,
            second_location,
            second_cards,
            registry,
        )
        if node.result_variable is not None:
            updated = set_effect_variable(updated, context, node.result_variable, change.changed)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                card_ids=tuple(move.card_id for move in change.card_moves),
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, RearrangeNode):
        color = _color_variable(updated, context, node.color_variable)
        if color is None:
            return updated, (), EffectStatus.COMPLETE
        order = _card_tuple_variable(updated, context, node.order_variable)
        updated, change = rearrange_stack(
            updated, resolve_player(node.player, context, updated), color, order, registry
        )
        if node.result_variable is not None:
            updated = set_effect_variable(updated, context, node.result_variable, change.changed)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                card_ids=order,
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, SplayNode):
        color = node.color
        if color is None:
            assert node.color_variable is not None
            color = _color_variable(updated, context, node.color_variable)
        direction = node.direction
        if direction is None:
            assert node.direction_variable is not None
            raw_direction = _text_variable(updated, context, node.direction_variable)
            direction = SplayDirection(raw_direction) if raw_direction is not None else None
        if color is None or direction is None:
            return updated, (), EffectStatus.COMPLETE
        updated, change = set_splay(
            updated,
            resolve_player(node.player, context, updated),
            color,
            direction,
            registry,
        )
        if node.result_variable is not None:
            updated = set_effect_variable(updated, context, node.result_variable, change.changed)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, RemoveAllPlayCardsNode):
        updated, change = _remove_all_play_cards(updated, registry)
        if node.result_variable is not None:
            updated = set_effect_variable(updated, context, node.result_variable, change.changed)
        if change.changed:
            updated, event = _event(
                updated,
                context,
                EffectEventKind.CHANGE,
                change=change,
                card_ids=tuple(move.card_id for move in change.card_moves),
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, NoOpNode):
        pass
    else:
        raise EffectInvariantError(f"node {type(node).__name__} is not an atomic leaf")
    return updated, tuple(events), EffectStatus.COMPLETE


def _choice_cards(
    state: GameState, context: EffectContext, node: ChoiceNode, registry: CardRegistry
) -> tuple[CardId, ...]:
    assert node.cards is not None
    return select_cards(state, context, node.cards, registry)


def _selected_cards(
    state: GameState, context: EffectContext, node: ChoiceNode
) -> tuple[CardId, ...]:
    return _card_tuple_variable(state, context, node.result_variable)


def _return_orders(
    cards: tuple[CardId, ...], registry: CardRegistry
) -> tuple[tuple[CardId, ...], ...]:
    groups: dict[int, tuple[CardId, ...]] = {}
    for age in sorted({registry.card(card_id).age for card_id in cards}):
        groups[age] = tuple(card_id for card_id in cards if registry.card(card_id).age == age)
    choices = tuple(tuple(itertools.permutations(groups[age])) for age in groups)
    return tuple(
        tuple(card for group in combination for card in group)
        for combination in itertools.product(*choices)
    )


def _order_options(
    cards: tuple[CardId, ...], node: ChoiceNode, registry: CardRegistry
) -> tuple[tuple[CardId, ...], ...]:
    if node.only_effective_return_order:
        return _return_orders(cards, registry)
    return tuple(itertools.permutations(cards))


def _choice_actions(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
) -> tuple[SemanticAction, ...]:
    decision_id = state.next_decision_id
    actions: list[SemanticAction] = []
    if node.choice_kind is ChoiceKind.CARD:
        actions.extend(
            ChooseCardAction(decision_id, card)
            for card in _choice_cards(state, context, node, registry)
        )
    elif node.choice_kind is ChoiceKind.BOUNDED_CARDS:
        selected = _selected_cards(state, context, node)
        remaining = (
            ()
            if len(selected) >= node.maximum
            else tuple(
                card
                for card in _choice_cards(state, context, node, registry)
                if card not in selected
            )
        )
        actions.extend(ChooseCardAction(decision_id, card) for card in remaining)
        if len(selected) >= node.minimum:
            actions.append(FinishSelectionAction(decision_id))
    elif node.choice_kind is ChoiceKind.COLOR:
        colors = node.colors
        if node.minimum_stack_size:
            target = node.target_player or node.chooser
            player = state.player(resolve_player(target, context, state))
            colors = tuple(
                color
                for color in colors
                if len(player.board.stack(color).cards) >= node.minimum_stack_size
            )
        actions.extend(ChooseColorAction(decision_id, color) for color in colors)
    elif node.choice_kind is ChoiceKind.PLAYER:
        resolved = tuple(resolve_player(reference, context, state) for reference in node.players)
        players = tuple(dict.fromkeys(resolved))
        actions.extend(ChoosePlayerAction(decision_id, player) for player in players)
    elif node.choice_kind is ChoiceKind.VALUE:
        actions.extend(ChooseValueAction(decision_id, value) for value in node.values)
    elif node.choice_kind is ChoiceKind.SPLAY:
        actions.extend(ChooseSplayAction(decision_id, direction) for direction in node.directions)
    elif node.choice_kind is ChoiceKind.BRANCH:
        actions.extend(ChooseBranchAction(decision_id, branch) for branch in node.branches)
    else:
        cards = _choice_cards(state, context, node, registry)
        actions.extend(
            OrderCardsAction(decision_id, order) for order in _order_options(cards, node, registry)
        )
    if node.optional and node.choice_kind is not ChoiceKind.BOUNDED_CARDS:
        actions.append(DeclineAction(decision_id))
    return tuple(actions)


def _auto_choice(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
) -> GameState | None:
    actions = _choice_actions(state, context, node, registry)
    substantive = tuple(action for action in actions if not isinstance(action, DeclineAction))
    if node.choice_kind is ChoiceKind.BOUNDED_CARDS:
        selected = _selected_cards(state, context, node)
        if len(selected) >= node.maximum:
            return state
        remaining = tuple(
            card for card in _choice_cards(state, context, node, registry) if card not in selected
        )
        if not remaining:
            return set_effect_variable(
                state,
                context,
                node.result_variable,
                tuple(card.value for card in selected),
            )
        return None
    if not substantive:
        return set_effect_variable(state, context, node.result_variable, None)
    if node.choice_kind is ChoiceKind.ORDER_CARDS:
        orders = tuple(action for action in substantive if isinstance(action, OrderCardsAction))
        if len(orders) == 1:
            return set_effect_variable(
                state,
                context,
                node.result_variable,
                tuple(card.value for card in orders[0].card_ids),
            )
    return None


def current_effect_decision(
    state: GameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> Decision | None:
    """Build the current player-safe decision when the top frame is a choice."""

    registry = registry or load_card_registry()
    if not state.pending_effects:
        return None
    frame = state.pending_effects[-1]
    if frame.kind != _NODE_FRAME:
        return None
    program = programs.program(_program_id(frame))
    node = program.node(_node_id(frame))
    if not isinstance(node, ChoiceNode):
        return None
    context = frame_context(frame)
    legal_actions = _choice_actions(state, context, node, registry)
    if not legal_actions:
        return None
    chooser = resolve_player(node.chooser, context, state)
    return Decision(
        state.next_decision_id,
        DecisionKind.EFFECT_CHOICE,
        chooser,
        context.executor,
        observe(state, chooser, registry),
        legal_actions,
        DecisionSource(context.source_card_id, context.source_effect_id),
        context.dogma_activator,
        context.dogma_action_id,
    )


def _advance_program(
    state: GameState,
    frame: EffectFrameState,
    programs: EffectProgramRegistry,
) -> EffectResolution:
    program = programs.program(_program_id(frame))
    context = frame_context(frame)
    entries: tuple[ProgramEffect, ...] = tuple(
        effect
        for effect in program.effects
        if not (_non_demand_only(frame) and effect.demand)
        and (
            _selected_effect_ordinal(frame) == 0
            or effect.effect_id.ordinal == _selected_effect_ordinal(frame)
        )
    )
    updated = state
    if frame.step > 0:
        previous = entries[frame.step - 1]
        previous_context = context.for_effect(previous.effect_id, demand=previous.demand)
        updated = clear_effect_scope(updated, previous_context)
    if frame.step >= len(entries):
        changes = qualifying_change_count(updated, context)
        updated = _pop(updated)
        if _is_root_program(frame):
            updated = clear_effect_scope(updated, context)
            return EffectResolution(
                updated,
                EffectStatus.COMPLETE,
                qualifying_changes=changes,
            )
        if context.nested:
            updated = clear_effect_scope(updated, context)
        return EffectResolution(updated, EffectStatus.CONTINUE)
    effect = entries[frame.step]
    next_frame = replace(frame, step=frame.step + 1)
    updated = _replace_top(updated, next_frame)
    effect_context = context.for_effect(effect.effect_id, demand=effect.demand)
    updated = _push(updated, _node_frame(program.program_id, effect.root_node_id, effect_context))
    return EffectResolution(updated, EffectStatus.CONTINUE)


def _advance_node(
    state: GameState,
    frame: EffectFrameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry,
) -> EffectResolution:
    program_id = _program_id(frame)
    program = programs.program(program_id)
    node = program.node(_node_id(frame))
    context = frame_context(frame)

    if isinstance(node, ChoiceNode):
        if frame.step == 0:
            state = delete_effect_variable(state, context, node.result_variable)
            frame = replace(frame, step=1)
            state = _replace_top(state, frame)
        automatic = _auto_choice(state, context, node, registry)
        if automatic is not None:
            return EffectResolution(_pop(automatic), EffectStatus.CONTINUE)
        decision = current_effect_decision(state, programs, registry)
        if decision is None:
            raise EffectInvariantError(f"choice node {node.node_id} produced no legal action")
        return EffectResolution(state, EffectStatus.AWAIT_DECISION, decision)
    if isinstance(node, SequenceNode):
        if frame.step >= len(node.children):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = _push(updated, _node_frame(program_id, node.children[frame.step], context))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, ConditionNode):
        branch = (
            node.when_true
            if evaluate_predicate(state, context, node.predicate, registry)
            else node.when_false
        )
        updated = _pop(state)
        if branch is not None:
            updated = _push(updated, _node_frame(program_id, branch, context))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, RepeatNode):
        if frame.step == 0:
            updated = _replace_top(state, replace(frame, step=1))
            updated = _push(updated, _node_frame(program_id, node.body, context))
            return EffectResolution(updated, EffectStatus.CONTINUE)
        if not evaluate_predicate(state, context, node.repeat_if, registry):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        if frame.step >= node.maximum_iterations:
            raise EffectInvariantError(
                f"repeat node {node.node_id} exceeded {node.maximum_iterations} iterations"
            )
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = _push(updated, _node_frame(program_id, node.body, context))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, BatchNode):
        updated = state
        all_events: list[EffectEvent] = []
        atomic_group_id = state.next_event_id
        for child_id in node.children:
            child = program.node(child_id)
            updated, events, status = _execute_leaf(
                updated,
                context,
                child,
                registry,
                atomic_group_id=atomic_group_id,
            )
            all_events.extend(events)
            if status is EffectStatus.TERMINAL:
                changes = qualifying_change_count(updated, context)
                updated = replace(updated, effect_variables=())
                return EffectResolution(
                    updated,
                    status,
                    events=tuple(all_events),
                    qualifying_changes=changes,
                )
        return EffectResolution(_pop(updated), EffectStatus.CONTINUE, events=tuple(all_events))
    if isinstance(node, NestedNode):
        card_id = _card_variable(state, context, node.card_variable)
        updated = _pop(state)
        if card_id is None:
            return EffectResolution(updated, EffectStatus.CONTINUE)
        try:
            nested_program = programs.program_for_card(card_id)
        except KeyError as error:
            raise EffectInvariantError(
                f"nested card {card_id} has no registered effect program"
            ) from error
        root = _root_context(context)
        count = get_effect_variable(state, root, _NESTED_COUNT, 0)
        if not isinstance(count, int) or isinstance(count, bool):
            raise EffectInvariantError("serialized nested execution count is invalid")
        updated = set_effect_variable(updated, root, _NESTED_COUNT, count + 1)
        nested_context = context.for_nested(card_id, f"nested-{count + 1}")
        updated = _push(
            updated,
            _program_frame(nested_program, nested_context, non_demand_only=True),
        )
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, AbortDogmaNode):
        changes = qualifying_change_count(state, context)
        updated, event = _event(state, context, EffectEventKind.ABORT_DOGMA)
        updated = replace(updated, pending_effects=(), effect_variables=())
        return EffectResolution(
            updated,
            EffectStatus.ABORT_DOGMA,
            events=(event,),
            qualifying_changes=changes,
        )

    updated, events, status = _execute_leaf(state, context, node, registry)
    if status is EffectStatus.TERMINAL:
        changes = qualifying_change_count(updated, context)
        updated = replace(updated, effect_variables=())
        return EffectResolution(
            updated,
            status,
            events=events,
            qualifying_changes=changes,
        )
    return EffectResolution(_pop(updated), EffectStatus.CONTINUE, events=events)


def step_effect(
    state: GameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> EffectResolution:
    """Advance exactly one explicit frame operation, suitable for checkpointing."""

    registry = registry or load_card_registry()
    if state.phase is GamePhase.TERMINAL:
        terminal = replace(state, pending_effects=(), effect_variables=())
        return EffectResolution(terminal, EffectStatus.TERMINAL)
    if not state.pending_effects:
        return EffectResolution(state, EffectStatus.COMPLETE)
    frame = state.pending_effects[-1]
    if frame.kind not in {_PROGRAM_FRAME, _NODE_FRAME}:
        return EffectResolution(state, EffectStatus.COMPLETE)
    context = frame_context(frame)
    if frame.kind == _NODE_FRAME:
        program = programs.program(_program_id(frame))
        node = program.node(_node_id(frame))
        if isinstance(node, ChoiceNode) and frame.step > 0:
            return _advance_node(state, frame, programs, registry)
    state = _bump_step(state, context)
    if frame.kind == _PROGRAM_FRAME:
        return _advance_program(state, frame, programs)
    return _advance_node(state, frame, programs, registry)


def resume_effect(
    state: GameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> EffectResolution:
    """Run deterministically until a choice, terminal, complete, or abort boundary."""

    registry = registry or load_card_registry()
    updated = state
    events: list[EffectEvent] = []
    changes = 0
    while True:
        if updated.pending_effects and updated.pending_effects[-1].kind in {
            _PROGRAM_FRAME,
            _NODE_FRAME,
        }:
            changes = qualifying_change_count(updated, frame_context(updated.pending_effects[-1]))
        result = step_effect(updated, programs, registry)
        updated = result.state
        events.extend(result.events)
        changes = max(changes, result.qualifying_changes)
        if result.status is not EffectStatus.CONTINUE:
            return EffectResolution(
                updated,
                result.status,
                result.decision,
                tuple(events),
                changes,
            )


def _start_effect(
    state: GameState,
    program_id: str,
    context: EffectContext,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None,
    *,
    selected_effect_ordinal: int,
    pause_before_first_step: bool,
) -> EffectResolution:
    if state.phase is GamePhase.TERMINAL:
        raise EffectInvariantError("cannot start an effect in a terminal state")
    if any(frame.kind in {_PROGRAM_FRAME, _NODE_FRAME} for frame in state.pending_effects):
        raise EffectInvariantError("cannot start an effect while another WP4 effect is pending")
    scope_prefixes = (f"{context.scope}:", f"{context.scope}/")
    if any(variable.name.startswith(scope_prefixes) for variable in state.effect_variables):
        raise EffectInvariantError("effect context scope is already in use")
    program = programs.program(program_id)
    if program.source_card_id != context.source_card_id:
        raise EffectInvariantError("effect context source card does not match the program")
    if selected_effect_ordinal and all(
        effect.effect_id.ordinal != selected_effect_ordinal for effect in program.effects
    ):
        raise EffectInvariantError(
            f"program {program_id} has no effect ordinal {selected_effect_ordinal}"
        )
    started = replace(
        state,
        pending_effects=(
            *state.pending_effects,
            _program_frame(
                program,
                context,
                non_demand_only=False,
                selected_effect_ordinal=selected_effect_ordinal,
                root_program=True,
            ),
        ),
    )
    started = set_effect_variable(started, _root_context(context), _STEP_COUNT, 0)
    started = set_effect_variable(started, _root_context(context), _QUALIFYING_CHANGE_COUNT, 0)
    started = set_effect_variable(started, _root_context(context), _NESTED_COUNT, 0)
    if pause_before_first_step:
        return EffectResolution(started, EffectStatus.CONTINUE)
    return resume_effect(started, programs, registry)


def start_effect(
    state: GameState,
    program_id: str,
    context: EffectContext,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
    *,
    pause_before_first_step: bool = False,
) -> EffectResolution:
    """Install a root program frame and optionally run it to the next boundary."""

    return _start_effect(
        state,
        program_id,
        context,
        programs,
        registry,
        selected_effect_ordinal=0,
        pause_before_first_step=pause_before_first_step,
    )


def start_program_effect(
    state: GameState,
    program_id: str,
    effect_ordinal: int,
    context: EffectContext,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
    *,
    pause_before_first_step: bool = False,
) -> EffectResolution:
    """Start exactly one printed effect, allowing WP5 to order executors per effect."""

    return _start_effect(
        state,
        program_id,
        context,
        programs,
        registry,
        selected_effect_ordinal=effect_ordinal,
        pause_before_first_step=pause_before_first_step,
    )


def submit_effect_action(
    state: GameState,
    action: SemanticAction,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> EffectResolution:
    """Apply one legal semantic effect choice and resume deterministic execution."""

    registry = registry or load_card_registry()
    decision = current_effect_decision(state, programs, registry)
    if decision is None:
        raise EffectInvariantError("effect stack is not awaiting a decision")
    if action not in decision.legal_actions:
        raise IllegalEffectAction(action, decision)
    frame = state.pending_effects[-1]
    program = programs.program(_program_id(frame))
    node = program.node(_node_id(frame))
    if not isinstance(node, ChoiceNode):  # pragma: no cover - guarded by decision
        raise EffectInvariantError("top effect frame is not a choice node")
    context = frame_context(frame)
    updated = set_effect_variable(
        state,
        context,
        _LAST_CHOOSER,
        decision.chooser.value,
    )
    complete = True
    if isinstance(action, ChooseCardAction):
        if node.choice_kind is ChoiceKind.BOUNDED_CARDS:
            selected = _selected_cards(updated, context, node)
            selected = (*selected, action.card_id)
            updated = set_effect_variable(
                updated,
                context,
                node.result_variable,
                tuple(card.value for card in selected),
            )
            complete = len(selected) >= node.maximum
        else:
            updated = set_effect_variable(
                updated, context, node.result_variable, action.card_id.value
            )
    elif isinstance(action, FinishSelectionAction | DeclineAction):
        if isinstance(action, DeclineAction):
            updated = set_effect_variable(updated, context, node.result_variable, None)
    elif isinstance(action, ChooseColorAction):
        updated = set_effect_variable(updated, context, node.result_variable, action.color.value)
    elif isinstance(action, ChoosePlayerAction):
        updated = set_effect_variable(
            updated, context, node.result_variable, action.player_id.value
        )
    elif isinstance(action, ChooseValueAction):
        updated = set_effect_variable(updated, context, node.result_variable, action.value)
    elif isinstance(action, ChooseSplayAction):
        updated = set_effect_variable(
            updated, context, node.result_variable, action.direction.value
        )
    elif isinstance(action, ChooseBranchAction):
        updated = set_effect_variable(updated, context, node.result_variable, action.branch_id)
    elif isinstance(action, OrderCardsAction):
        updated = set_effect_variable(
            updated,
            context,
            node.result_variable,
            tuple(card.value for card in action.card_ids),
        )
    else:  # pragma: no cover - legal effect actions exhaust the choice kinds
        raise EffectInvariantError(f"unsupported effect choice action: {action.kind}")
    if complete:
        updated = _pop(updated)
    updated = replace(updated, next_decision_id=updated.next_decision_id + 1)
    return resume_effect(updated, programs, registry)
