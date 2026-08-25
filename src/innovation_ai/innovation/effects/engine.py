"""Explicit stack interpreter for resumable Innovation effects."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from innovation_ai.innovation.achievements import (
    AchievementCheckResult,
    LinkedRouteContext,
    check_atomic_boundary,
    claim_linked_route,
)
from innovation_ai.innovation.actions import (
    ChooseBranchAction,
    ChooseCardAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseValueAction,
    Decision,
    DecisionContext,
    DecisionKind,
    DecisionSource,
    DeclineAction,
    FinishSelectionAction,
    IncrementalSelectionKind,
    SemanticAction,
)
from innovation_ai.innovation.board import (
    immediately_beneath,
    score_value,
    top_cards,
    visible_icons,
    visible_icons_for_stack,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import GameObservation, observe
from innovation_ai.innovation.state import (
    PUBLIC_REVEALED_COLOR_PREFIX,
    ColorStack,
    EffectFrameState,
    GamePhase,
    GameState,
    TerminalState,
)
from innovation_ai.innovation.terminal import (
    apply_terminal,
    direct_card_effect_win,
    draw_beyond_age_ten_result,
    unique_lowest_result,
    unique_most_result,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import (
    CardLocation,
    ChangeKind,
    ChangeRecord,
    SplayChange,
    ZoneKind,
    clear_revealed_cards,
    clear_revealed_scope,
    draw_card,
    exchange_cards,
    mark_revealed,
    meld_card,
    move_card,
    rearrange_stack,
    remove_all_cards_in_play,
    remove_card,
    return_card,
    score_card,
    set_splay,
    tuck_card,
)

from .model import (
    DOGMA_FRAME,
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
    frozen_icon_counts,
    get_effect_variable,
    make_frame,
    set_effect_variable,
    variable_context,
)
from .program import (
    AbortDogmaNode,
    AllOrNoneNode,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    Cmp,
    CollectNode,
    ConditionNode,
    DrawAndMoveNode,
    DrawNode,
    EffectNode,
    EffectProgram,
    EffectProgramRegistry,
    ExchangeNode,
    Extreme,
    ExtremeScope,
    ForEachCardNode,
    KeepNode,
    LetNode,
    MovementKind,
    MovementResultMode,
    MoveNode,
    NestedNode,
    NoOpNode,
    OrderGroup,
    PlayerRef,
    PlayerRefKind,
    Predicate,
    PredicateKind,
    ProgramEffect,
    RearrangeNode,
    RemoveAllPlayCardsNode,
    RepeatNode,
    RevealColorNode,
    RevealNode,
    Rounding,
    SelectorRelationKind,
    SequenceNode,
    SplayNode,
    StackPosition,
    StopNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
    WinMetric,
    WinMode,
    WinNode,
)

_PROGRAM_FRAME = "effect-program"
_NODE_FRAME = "effect-node"
_LAST_CHOOSER = "last-chooser"
_STEP_COUNT = "step-count"
_QUALIFYING_CHANGE_COUNT = "qualifying-change-count"
_NESTED_COUNT = "nested-count"
_NESTED_DEPTH = "nested_depth"
_CANDIDATE = "_candidate"
_ORDER_PROGRESS = "_order-progress"
_HIDDEN_VALUE = "_hidden-value"
_TIMES_LIMIT = "_times-limit"
MAX_NESTED_DEPTH = 16
"""Explicit nesting budget so Robotics -> Software -> Robotics fails loudly (decision 6)."""


def other_player(player_id: PlayerId) -> PlayerId:
    """Return the opponent in the supported two-player game."""

    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def _known_card_ids(observation: GameObservation) -> frozenset[CardId]:
    """Return every exact card identity visible in one player-safe observation."""

    known: set[CardId] = set(observation.revealed_cards)
    for player in observation.players:
        known.update(player.hand.known_cards)
        known.update(player.score_pile.known_cards)
        for stack in player.board:
            if stack.top_card_id is not None:
                known.add(stack.top_card_id)
            known.update(card.card_id for card in stack.covered_cards if card.card_id is not None)
    return frozenset(known)


def _require_exact_choice_visibility(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    cards: tuple[CardId, ...],
    registry: CardRegistry,
) -> None:
    """Reject exact-card actions that would disclose identities the chooser cannot inspect.

    Card modules must use ``HIDDEN_CARD`` for a private hand/score choice. A normal ``CARD`` node
    is legal only when all offered IDs are already present in the chooser's observation, including
    Collaboration-style cards that were explicitly revealed first.
    """

    chooser = resolve_player(node.chooser, context, state)
    visible = _known_card_ids(observe(state, chooser, registry))
    hidden = tuple(card_id for card_id in cards if card_id not in visible)
    if hidden:
        raise EffectInvariantError(
            f"choice node {node.node_id} exposes hidden card identities; use hidden-card choice"
        )


def resolve_player(
    reference: PlayerRef,
    context: EffectContext,
    state: GameState | None = None,
) -> PlayerId:
    """Resolve a declarative player reference to exactly one player."""

    if reference.is_multi:
        raise EffectInvariantError(
            f"{reference.kind} denotes several players and needs a set-valued caller"
        )
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


def resolve_players(
    reference: PlayerRef,
    context: EffectContext,
    state: GameState | None = None,
) -> tuple[PlayerId, ...]:
    """Resolve a player reference to a canonically ordered set of players."""

    if reference.kind is PlayerRefKind.ALL:
        return tuple(PlayerId)
    if reference.kind is PlayerRefKind.ALL_OTHER:
        return tuple(player_id for player_id in PlayerId if player_id is not context.executor)
    return (resolve_player(reference, context, state),)


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


def _value_filter_matches(candidate: int, target: int, comparator: Cmp) -> bool:
    if comparator is Cmp.EQ:
        return candidate == target
    if comparator is Cmp.NE:
        return candidate != target
    if comparator is Cmp.LT:
        return candidate < target
    if comparator is Cmp.LE:
        return candidate <= target
    if comparator is Cmp.GT:
        return candidate > target
    return candidate >= target


def _stack_positions(
    state: GameState,
    context: EffectContext,
    selector: CardSelector,
    stack: ColorStack,
) -> tuple[CardId, ...]:
    if selector.position is StackPosition.ANY:
        return stack.cards
    if selector.position is StackPosition.TOP:
        return (stack.top,) if stack.top is not None else ()
    if selector.position is StackPosition.BOTTOM:
        return (stack.bottom,) if stack.bottom is not None else ()
    if selector.position is StackPosition.BENEATH_TOP:
        return (stack.beneath_top,) if stack.beneath_top is not None else ()
    assert selector.position_variable is not None
    anchor = _card_variable(state, context, selector.position_variable)
    if anchor is None or anchor not in stack.cards:
        return ()
    beneath = immediately_beneath(stack, anchor)
    return (beneath,) if beneath is not None else ()


def select_cards(
    state: GameState,
    context: EffectContext,
    selector: CardSelector,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
    *,
    for_choice: bool = False,
) -> tuple[CardId, ...]:
    """Evaluate a deterministic card selector without exposing callbacks.

    Filters compose in the fixed order documented on :class:`CardSelector`, so the same selector
    always denotes the same set for a given state.

    ``for_choice`` distinguishes the two consumers of a selector. A choice node enumerates
    candidates and therefore wants every tied extreme so the chooser can break the tie. A direct
    consumer such as a movement has no chooser, so an ``ONE_TIED`` extreme collapses to the lowest
    stable card ID - decision 13's deterministic fallback when nobody can choose.
    """

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
        cards = ()
        for player_id in resolve_players(selector.player, context, state):
            player = state.player(player_id)
            if selector.kind is CardSelectorKind.HAND:
                cards += player.hand
            elif selector.kind is CardSelectorKind.SCORE:
                cards += player.score_pile
            elif selector.kind is CardSelectorKind.BOARD_STACK:
                stack = player.board.stack(_resolve_color(state, context, selector))
                cards += _stack_positions(state, context, selector, stack)
            elif selector.kind is CardSelectorKind.BOARD_ALL:
                cards += tuple(
                    card_id
                    for color in Color
                    for card_id in _stack_positions(
                        state, context, selector, player.board.stack(color)
                    )
                )
            elif selector.kind is CardSelectorKind.TOP_CARDS:
                cards += top_cards(player.board)
            else:  # pragma: no cover - exhaustive guard
                raise EffectInvariantError(f"unhandled card selector: {selector.kind}")

    if selector.colors:
        allowed = set(selector.colors)
        cards = tuple(card_id for card_id in cards if registry.card(card_id).color in allowed)
    if selector.exclude_colors:
        blocked = set(selector.exclude_colors)
        cards = tuple(card_id for card_id in cards if registry.card(card_id).color not in blocked)
    if selector.icon is not None:
        cards = tuple(
            card_id for card_id in cards if selector.icon in registry.card(card_id).functional_icons
        )
    if selector.without_icon is not None:
        cards = tuple(
            card_id
            for card_id in cards
            if selector.without_icon not in registry.card(card_id).functional_icons
        )
    if selector.value is not None or selector.value_expr is not None:
        target = (
            selector.value
            if selector.value is not None
            else resolve_value(
                state,
                context,
                cast(ValueRef, selector.value_expr),
                registry,
                programs,
            )
        )
        assert target is not None
        target += selector.value_offset
        cards = tuple(
            card_id
            for card_id in cards
            if _value_filter_matches(registry.card(card_id).age, target, selector.value_cmp)
        )
    if selector.relation is not None:
        reference = select_cards(state, context, selector.relation.reference, registry, programs)
        if selector.relation.kind is SelectorRelationKind.SAME_COLOR_AS_ANY:
            colors = {registry.card(card_id).color for card_id in reference}
            cards = tuple(card_id for card_id in cards if registry.card(card_id).color in colors)
        elif selector.relation.kind is SelectorRelationKind.DIFFERENT_COLOR_FROM_ALL:
            colors = {registry.card(card_id).color for card_id in reference}
            cards = tuple(
                card_id for card_id in cards if registry.card(card_id).color not in colors
            )
        else:
            values = {registry.card(card_id).age for card_id in reference}
            cards = tuple(card_id for card_id in cards if registry.card(card_id).age in values)
    if selector.predicate is not None:
        if programs is None:
            raise EffectInvariantError(
                "a named selector predicate requires the effect program registry"
            )
        cards = tuple(
            card_id
            for card_id in cards
            if _named_card_predicate(
                state, context, registry, programs, selector.predicate, card_id
            )
        )
    if selector.exclude_variable is not None:
        raw_excluded = get_effect_variable(state, context, selector.exclude_variable)
        if raw_excluded is None:
            excluded: set[CardId] = set()
        elif isinstance(raw_excluded, str):
            excluded = {CardId(raw_excluded)}
        elif isinstance(raw_excluded, tuple) and all(
            isinstance(item, str) for item in raw_excluded
        ):
            excluded = {CardId(item) for item in cast(tuple[str, ...], raw_excluded)}
        else:
            raise EffectInvariantError(
                f"variable {selector.exclude_variable!r} is not a card selection"
            )
        cards = tuple(card_id for card_id in cards if card_id not in excluded)
    if selector.exclude_source_card:
        cards = tuple(card_id for card_id in cards if card_id != context.source_card_id)
    if selector.extreme is not None and cards:
        ages = tuple(registry.card(card_id).age for card_id in cards)
        extreme_value = max(ages) if selector.extreme is Extreme.HIGHEST else min(ages)
        cards = tuple(card_id for card_id in cards if registry.card(card_id).age == extreme_value)
        if selector.extreme_scope is ExtremeScope.ONE_TIED and not for_choice and len(cards) > 1:
            cards = (min(cards, key=str),)
    return cards


def _named_card_predicate(
    state: GameState,
    context: EffectContext,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
    name: str,
    card_id: CardId,
) -> bool:
    """Evaluate a named selector predicate for one candidate card.

    The candidate is passed through the reserved ``_candidate`` scoped variable so the card
    module's callable stays a pure ``(state, context, registry)`` function.
    """

    scoped = set_effect_variable(state, context, _CANDIDATE, card_id.value)
    return programs.named_predicate(context.source_card_id, name)(scoped, context, registry)


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


def _divide(raw: int, per: int, rounding: Rounding) -> int:
    if rounding is Rounding.CEIL:
        return -((-raw) // per)
    return raw // per


def resolve_value(
    state: GameState,
    context: EffectContext,
    reference: ValueRef,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> int:
    """Evaluate a small integer expression from state and scoped serializable values.

    Per rules decision 17 the caller decides *when* to evaluate; this function is a pure read of
    the state it is given. Nodes that own a quantity evaluate it once on entry, and ``Repeat`` or
    an explicit dogma repeat reenters the node and therefore reevaluates.
    """

    registry = registry or load_card_registry()
    raw: int
    if reference.kind is ValueRefKind.LITERAL:
        assert reference.value is not None
        raw = reference.value
    elif reference.kind is ValueRefKind.VARIABLE:
        assert reference.variable is not None
        value = get_effect_variable(state, context, reference.variable)
        if not isinstance(value, int) or isinstance(value, bool):
            raise EffectInvariantError(f"variable {reference.variable!r} is not an integer")
        raw = value
    elif reference.kind is ValueRefKind.COUNT_CARDS:
        assert reference.variable is not None
        value = get_effect_variable(state, context, reference.variable)
        if value is None:
            raw = 0
        elif isinstance(value, str):
            raw = 1
        elif isinstance(value, tuple):
            raw = len(value)
        else:
            raise EffectInvariantError(f"variable {reference.variable!r} is not countable")
    elif reference.kind is ValueRefKind.COUNT_SELECTOR:
        assert reference.selector is not None
        raw = len(select_cards(state, context, reference.selector, registry, programs))
    elif reference.kind is ValueRefKind.ICON_COUNT:
        assert reference.icon is not None and reference.player is not None
        player_id = resolve_player(reference.player, context, state)
        raw = visible_icons(state.player(player_id).board, registry)[reference.icon]
    elif reference.kind is ValueRefKind.COLORS_WITH_ICON:
        assert reference.icon is not None and reference.player is not None
        board = state.player(resolve_player(reference.player, context, state)).board
        raw = sum(
            1
            for stack in board.stacks
            if visible_icons_for_stack(stack, registry)[reference.icon] >= 1
        )
    elif reference.kind is ValueRefKind.COLORS_PRESENT_ONLY_HERE:
        assert reference.player is not None
        player_id = resolve_player(reference.player, context, state)
        others = tuple(other for other in PlayerId if other is not player_id)
        raw = sum(
            1
            for color in Color
            if state.player(player_id).board.stack(color).cards
            and all(not state.player(other).board.stack(color).cards for other in others)
        )
    elif reference.kind is ValueRefKind.COLORS_SPLAYED:
        assert reference.player is not None and reference.direction is not None
        board = state.player(resolve_player(reference.player, context, state)).board
        raw = sum(1 for stack in board.stacks if stack.splay is reference.direction)
    elif reference.kind is ValueRefKind.DISTINCT_VALUES:
        assert reference.variable is not None
        cards = _card_tuple_variable(state, context, reference.variable)
        raw = len({registry.card(card_id).age for card_id in cards})
    elif reference.kind is ValueRefKind.CARD_VALUE:
        assert reference.variable is not None
        card_id = _card_variable(state, context, reference.variable)
        raw = 0 if card_id is None else registry.card(card_id).age
    elif reference.kind is ValueRefKind.SELECTOR_EXTREME:
        assert reference.selector is not None and reference.extreme is not None
        cards = select_cards(state, context, reference.selector, registry, programs)
        values = tuple(registry.card(card_id).age for card_id in cards)
        if not values:
            raw = 0
        else:
            raw = max(values) if reference.extreme is Extreme.HIGHEST else min(values)
    elif reference.kind is ValueRefKind.SCORE:
        assert reference.player is not None
        raw = score_value(state.player(resolve_player(reference.player, context, state)), registry)
    elif reference.kind is ValueRefKind.ACHIEVEMENT_COUNT:
        assert reference.player is not None
        raw = state.player(resolve_player(reference.player, context, state)).achievement_count
    else:
        assert reference.name is not None
        if programs is None:
            raise EffectInvariantError("a named quantity requires the effect program registry")
        raw = programs.named_value(context.source_card_id, reference.name)(state, context, registry)
    return _divide(raw, reference.per, reference.rounding) + reference.offset


def _compare(left: int, comparator: Cmp, right: int) -> bool:
    if comparator is Cmp.EQ:
        return left == right
    if comparator is Cmp.NE:
        return left != right
    if comparator is Cmp.LT:
        return left < right
    if comparator is Cmp.LE:
        return left <= right
    if comparator is Cmp.GT:
        return left > right
    return left >= right


def evaluate_predicate(
    state: GameState,
    context: EffectContext,
    predicate: Predicate,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> bool:
    """Evaluate an explicit condition against current state and scoped variables.

    Universal card tests are true for an empty candidate set, which is rules decision 10
    implemented exactly once for every card that needs it.
    """

    if predicate.kind is PredicateKind.NOT:
        assert predicate.operand is not None
        return not evaluate_predicate(state, context, predicate.operand, registry, programs)
    if predicate.kind is PredicateKind.COUNT_CMP:
        assert predicate.left is not None and predicate.right is not None
        return _compare(
            resolve_value(state, context, predicate.left, registry, programs),
            predicate.comparator,
            resolve_value(state, context, predicate.right, registry, programs),
        )
    if predicate.kind in {
        PredicateKind.ALL_CARDS_MATCH,
        PredicateKind.ANY_CARDS_MATCH,
        PredicateKind.SELECTOR_NON_EMPTY,
    }:
        assert predicate.cards is not None
        candidates = select_cards(state, context, predicate.cards, registry, programs)
        if predicate.kind is PredicateKind.SELECTOR_NON_EMPTY:
            return bool(candidates)
        assert predicate.match is not None
        matching = set(select_cards(state, context, predicate.match, registry, programs))
        if predicate.kind is PredicateKind.ALL_CARDS_MATCH:
            return all(card_id in matching for card_id in candidates)
        return any(card_id in matching for card_id in candidates)
    if predicate.kind is PredicateKind.NAMED:
        assert predicate.name is not None
        if programs is None:
            raise EffectInvariantError("a named predicate requires the effect program registry")
        return programs.named_predicate(context.source_card_id, predicate.name)(
            state, context, registry
        )

    assert predicate.variable is not None
    scoped_context = variable_context(context, predicate.variable_scope)
    value = get_effect_variable(state, scoped_context, predicate.variable)
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
    if predicate.kind is PredicateKind.CARD_COLOR_IS:
        assert predicate.color is not None
        return card.color is predicate.color
    allowed: tuple[Color, ...]
    if predicate.colors_variable is not None:
        raw = get_effect_variable(state, context, predicate.colors_variable, ())
        if isinstance(raw, str):
            allowed = (Color(raw),)
        elif isinstance(raw, tuple) and all(isinstance(item, str) for item in raw):
            allowed = tuple(Color(item) for item in cast(tuple[str, ...], raw))
        else:
            raise EffectInvariantError(
                f"variable {predicate.colors_variable!r} is not a colour selection"
            )
    else:
        allowed = predicate.colors
    return card.color in allowed


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


def _nested_depth(frame: EffectFrameState) -> int:
    value = frame_value(frame, _NESTED_DEPTH, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EffectInvariantError("effect frame has an invalid nested depth")
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


def _node_frame(
    program_id: str, node_id: str, context: EffectContext, *, nested_depth: int = 0
) -> EffectFrameState:
    return make_frame(
        _NODE_FRAME,
        context,
        program_id=program_id,
        node_id=node_id,
        **{_NESTED_DEPTH: nested_depth},
    )


def _program_frame(
    program: EffectProgram,
    context: EffectContext,
    *,
    non_demand_only: bool,
    selected_effect_ordinal: int = 0,
    root_program: bool = False,
    nested_depth: int = 0,
) -> EffectFrameState:
    return make_frame(
        _PROGRAM_FRAME,
        context,
        program_id=program.program_id,
        non_demand_only=non_demand_only,
        selected_effect_ordinal=selected_effect_ordinal,
        root_program=root_program,
        **{_NESTED_DEPTH: nested_depth},
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
    revealed_colors: tuple[Color, ...] = (),
    achievement_player: PlayerId | None = None,
    achievement_id: NormalAchievementId | SpecialAchievementId | None = None,
    atomic_group_id: int | None = None,
) -> tuple[GameState, EffectEvent]:
    causal = _effective_context(state, context)
    # Decision 2: any player-facing gameplay change qualifies, which includes a reveal and an
    # achievement claim. Frame progress and bookkeeping never do.
    qualifying = kind in {EffectEventKind.REVEAL, EffectEventKind.ACHIEVEMENT} or (
        kind is EffectEventKind.CHANGE and change is not None and change.changed
    )
    if qualifying and state.phase is not GamePhase.TERMINAL:
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
        change=change,
        card_ids=card_ids,
        revealed_colors=revealed_colors,
        achievement_player=achievement_player,
        achievement_id=achievement_id,
        atomic_group_id=group_id,
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


def _movement_order(
    state: GameState,
    context: EffectContext,
    node: MoveNode,
    cards: tuple[CardId, ...],
) -> tuple[CardId, ...]:
    """Apply an explicitly chosen movement order, defaulting to stable card-ID order.

    Rules decision 16 separates unordered subset selection from movement order: the subset is
    canonical, and only the movement uses a chosen order when one was requested.
    """

    if node.order_variable is None:
        return tuple(sorted(cards, key=str))
    requested = _card_tuple_variable(state, context, node.order_variable)
    if not requested:
        return tuple(sorted(cards, key=str))
    if set(requested) != set(cards):
        raise EffectInvariantError(
            f"movement order variable {node.order_variable!r} does not cover the moved cards"
        )
    return requested


def _movement_change(
    state: GameState,
    context: EffectContext,
    node: MoveNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> tuple[GameState, ChangeRecord, tuple[CardId, ...]]:
    selected = select_cards(state, context, node.cards, registry, programs)
    cards = _movement_order(state, context, node, selected)
    changes: list[ChangeRecord] = []
    moved: list[CardId] = []
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
            if node.destination_zone is ZoneKind.HAND:
                destination = CardLocation.hand(player)
            elif node.destination_zone is ZoneKind.SCORE:
                destination = CardLocation.score(player)
            else:
                # Decision 14: a board transfer lands atop the matching stack and adopts its
                # splay, which the shared zone primitive already does.
                destination = CardLocation.board(player, registry.card(card_id).color)
            updated, change = move_card(
                updated,
                card_id,
                destination,
                registry,
                kind=ChangeKind.TRANSFER,
            )
        if change.changed:
            moved.append(card_id)
        changes.append(change)
    return (
        updated,
        _combine_changes(ChangeKind(node.movement.value), tuple(changes)),
        tuple(moved),
    )


def _execute_leaf(
    state: GameState,
    context: EffectContext,
    node: EffectNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
    *,
    atomic_group_id: int | None = None,
) -> tuple[GameState, tuple[EffectEvent, ...], EffectStatus]:
    events: list[EffectEvent] = []
    updated = state
    if isinstance(node, DrawNode):
        player = resolve_player(node.player, context, updated)
        updated, change, result = draw_card(
            updated,
            resolve_value(updated, context, node.requested_age, registry, programs),
            player,
            registry,
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
            terminal = draw_beyond_age_ten_result(updated, registry)
            updated = apply_terminal(updated, terminal)
            return updated, tuple(events), EffectStatus.TERMINAL
    elif isinstance(node, RevealNode):
        cards = select_cards(updated, context, node.cards, registry, programs)
        if cards:
            # Decision 18: every physically face-up card gets a marker, but only identities that
            # were not already public to both players create a qualifying reveal event.
            public_before = {
                player_id: _known_card_ids(observe(updated, player_id, registry))
                for player_id in PlayerId
            }
            newly_public = tuple(
                card_id
                for card_id in cards
                if any(card_id not in public_before[player_id] for player_id in PlayerId)
            )
            updated, _ = mark_revealed(updated, cards, context.scope)
            if newly_public:
                updated, event = _event(
                    updated,
                    context,
                    EffectEventKind.REVEAL,
                    card_ids=newly_public,
                    atomic_group_id=atomic_group_id,
                )
                events.append(event)
    elif isinstance(node, RevealColorNode):
        color = _color_variable(updated, context, node.color_variable)
        if color is not None:
            updated = set_effect_variable(
                updated,
                context,
                f"{PUBLIC_REVEALED_COLOR_PREFIX}{node.node_id}",
                color.value,
            )
            updated, event = _event(
                updated,
                context,
                EffectEventKind.REVEAL,
                revealed_colors=(color,),
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, KeepNode):
        cards = select_cards(updated, context, node.cards, registry, programs)
        if cards:
            # Keeping a revealed card puts it back into a hidden zone in place.
            updated = clear_revealed_cards(updated, cards)
            updated, event = _event(
                updated,
                context,
                EffectEventKind.KEEP,
                card_ids=cards,
                atomic_group_id=atomic_group_id,
            )
            events.append(event)
    elif isinstance(node, MoveNode):
        updated, change, moved = _movement_change(updated, context, node, registry, programs)
        if node.result_variable is not None:
            result_context = variable_context(context, node.result_scope)
            movement_result = change.changed
            if node.result_mode is MovementResultMode.ANY:
                previous = get_effect_variable(updated, result_context, node.result_variable, False)
                if not isinstance(previous, bool):
                    raise EffectInvariantError(
                        f"movement result {node.result_variable!r} is not boolean"
                    )
                movement_result = previous or movement_result
            updated = set_effect_variable(
                updated, result_context, node.result_variable, movement_result
            )
        if node.moved_variable is not None:
            updated = set_effect_variable(
                updated,
                context,
                node.moved_variable,
                tuple(card_id.value for card_id in moved),
            )
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
    elif isinstance(node, DrawAndMoveNode):
        player = resolve_player(node.player, context, updated)
        total = max(0, resolve_value(updated, context, node.count, registry, programs))
        if total > node.maximum_iterations:
            raise EffectInvariantError(
                f"draw-and-move node {node.node_id} requested {total} iterations, "
                f"exceeding {node.maximum_iterations}"
            )
        group_id = atomic_group_id if atomic_group_id is not None else updated.next_event_id
        for _ in range(total):
            updated, draw_change, draw_result = draw_card(
                updated,
                resolve_value(updated, context, node.requested_age, registry, programs),
                player,
                registry,
            )
            if draw_change.changed:
                updated, event = _event(
                    updated,
                    context,
                    EffectEventKind.CHANGE,
                    change=draw_change,
                    card_ids=(draw_result.card_id,) if draw_result.card_id is not None else (),
                    atomic_group_id=group_id,
                )
                events.append(event)
            if draw_result.beyond_age_ten:
                terminal = draw_beyond_age_ten_result(updated, registry)
                updated = apply_terminal(updated, terminal)
                return updated, tuple(events), EffectStatus.TERMINAL
            assert draw_result.card_id is not None
            if node.movement is MovementKind.MELD:
                updated, move_change = meld_card(updated, player, draw_result.card_id, registry)
            elif node.movement is MovementKind.TUCK:
                updated, move_change = tuck_card(updated, player, draw_result.card_id, registry)
            else:
                updated, move_change = score_card(updated, player, draw_result.card_id, registry)
            if move_change.changed:
                updated, event = _event(
                    updated,
                    context,
                    EffectEventKind.CHANGE,
                    change=move_change,
                    card_ids=(draw_result.card_id,),
                    atomic_group_id=group_id,
                )
                events.append(event)
    elif isinstance(node, ExchangeNode):
        first_location = _selector_location(updated, context, node.first)
        second_location = _selector_location(updated, context, node.second)
        first_cards = select_cards(updated, context, node.first, registry, programs)
        second_cards = select_cards(updated, context, node.second, registry, programs)
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
        updated, change = remove_all_cards_in_play(updated, registry)
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
    elif isinstance(node, LetNode):
        result_context = variable_context(context, node.result_scope)
        if node.value is not None:
            updated = set_effect_variable(
                updated,
                result_context,
                node.result_variable,
                resolve_value(updated, context, node.value, registry, programs),
            )
        elif node.color_of is not None:
            source = _card_variable(updated, context, node.color_of)
            updated = set_effect_variable(
                updated,
                result_context,
                node.result_variable,
                None if source is None else registry.card(source).color.value,
            )
        else:
            assert node.cards is not None
            selected = select_cards(updated, context, node.cards, registry, programs)
            updated = set_effect_variable(
                updated,
                result_context,
                node.result_variable,
                tuple(card_id.value for card_id in selected),
            )
    elif isinstance(node, ClaimAchievementNode):
        updated, events_from_claim, status = _claim_achievement(
            updated, context, node, registry, atomic_group_id=atomic_group_id
        )
        events.extend(events_from_claim)
        if status is not EffectStatus.COMPLETE:
            return updated, tuple(events), status
    elif isinstance(node, WinNode):
        win = _win_result(updated, context, node, registry)
        if win is None:
            # Rules section 11: a tie ignores the entire win effect and play continues.
            return updated, tuple(events), EffectStatus.COMPLETE
        updated = apply_terminal(updated, win)
        return updated, tuple(events), EffectStatus.TERMINAL
    elif isinstance(node, NoOpNode):
        pass
    else:
        raise EffectInvariantError(f"node {type(node).__name__} is not an atomic leaf")
    return updated, tuple(events), EffectStatus.COMPLETE


def _claim_achievement(
    state: GameState,
    context: EffectContext,
    node: ClaimAchievementNode,
    registry: CardRegistry,
    *,
    atomic_group_id: int | None = None,
) -> tuple[GameState, tuple[EffectEvent, ...], EffectStatus]:
    """Claim a linked special-achievement route and honour an immediate sixth-achievement win."""

    melded = 0
    if node.melded_count_variable is not None:
        raw = get_effect_variable(state, context, node.melded_count_variable, 0)
        if isinstance(raw, tuple):
            melded = len(raw)
        elif isinstance(raw, int) and not isinstance(raw, bool):
            melded = raw
    player_id = resolve_player(node.player, context, state)
    result = claim_linked_route(
        state,
        player_id,
        node.achievement_id,
        registry,
        context=LinkedRouteContext(melded_card_count=melded),
    )
    updated = result.state
    if node.result_variable is not None and result.terminal is None:
        updated = set_effect_variable(updated, context, node.result_variable, result.changed)
    events: list[EffectEvent] = []
    for _claim in result.claims:
        # Decision 2 counts each achievement claim as a player-facing change, so it emits a
        # qualifying event and can justify the sharing bonus by itself.
        updated, event = _event(
            updated,
            context,
            EffectEventKind.ACHIEVEMENT,
            achievement_player=_claim.player_id,
            achievement_id=_claim.achievement_id,
            atomic_group_id=atomic_group_id,
        )
        events.append(event)
    if result.terminal is not None:
        return updated, tuple(events), EffectStatus.TERMINAL
    return updated, tuple(events), EffectStatus.COMPLETE


def _win_result(
    state: GameState,
    context: EffectContext,
    node: WinNode,
    registry: CardRegistry,
) -> TerminalState | None:
    if node.mode is WinMode.EXECUTOR:
        return direct_card_effect_win(resolve_player(node.player, context, state))
    assert node.metric is not None
    if node.metric is WinMetric.SCORE:
        counts = {
            player_id: score_value(state.player(player_id), registry) for player_id in PlayerId
        }
    elif node.metric is WinMetric.ACHIEVEMENTS:
        counts = {player_id: state.player(player_id).achievement_count for player_id in PlayerId}
    else:
        assert node.icon is not None
        counts = {
            player_id: visible_icons(state.player(player_id).board, registry)[node.icon]
            for player_id in PlayerId
        }
    if node.extreme is Extreme.HIGHEST:
        return unique_most_result(counts)
    return unique_lowest_result(counts)


def _choice_cards(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> tuple[CardId, ...]:
    assert node.cards is not None
    return select_cards(state, context, node.cards, registry, programs, for_choice=True)


def _selected_cards(
    state: GameState, context: EffectContext, node: ChoiceNode
) -> tuple[CardId, ...]:
    return _card_tuple_variable(state, context, node.result_variable)


def _order_progress(
    state: GameState, context: EffectContext, node: ChoiceNode
) -> tuple[CardId, ...]:
    return _card_tuple_variable(state, context, f"{node.result_variable}{_ORDER_PROGRESS}")


def _order_group_key(card_id: CardId, node: ChoiceNode, registry: CardRegistry) -> str:
    if node.order_group is OrderGroup.AGE:
        return f"age-{registry.card(card_id).age}"
    if node.order_group is OrderGroup.COLOR:
        return registry.card(card_id).color.value
    return "all"


def _order_needs_decision(
    cards: tuple[CardId, ...], node: ChoiceNode, registry: CardRegistry
) -> bool:
    """Whether any pair of ordered cards can actually distinguish two authoritative states."""

    if len(cards) < 2:
        return False
    groups: dict[str, int] = {}
    for card_id in cards:
        key = _order_group_key(card_id, node, registry)
        groups[key] = groups.get(key, 0) + 1
    return any(count >= 2 for count in groups.values())


def _canonical_order(
    cards: tuple[CardId, ...], node: ChoiceNode, registry: CardRegistry
) -> tuple[CardId, ...]:
    return tuple(
        sorted(
            cards, key=lambda card_id: (_order_group_key(card_id, node, registry), card_id.value)
        )
    )


def _order_candidates(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> tuple[CardId, ...]:
    """Return the cards still choosable as the next ordered card.

    Ordering is incremental, so ``k`` cards cost ``k`` decisions of at most ``k`` actions rather
    than enumerating ``k!`` permutations. Within an order-group boundary only the current group's
    remaining cards are offered, because cross-group order is unobservable.
    """

    cards = _choice_cards(state, context, node, registry, programs)
    chosen = _order_progress(state, context, node)
    remaining = tuple(card_id for card_id in cards if card_id not in chosen)
    if not remaining:
        return ()
    if node.order_group is OrderGroup.ALL:
        return remaining
    groups: list[str] = []
    for card_id in _canonical_order(cards, node, registry):
        key = _order_group_key(card_id, node, registry)
        if key not in groups:
            groups.append(key)
    for key in groups:
        pending = tuple(
            card_id for card_id in remaining if _order_group_key(card_id, node, registry) == key
        )
        if pending:
            return pending
    return ()  # pragma: no cover - covered by the remaining check above


def _hidden_values(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> tuple[int, ...]:
    cards = _choice_cards(state, context, node, registry, programs)
    return tuple(sorted({registry.card(card_id).age for card_id in cards}))


def _hidden_stage(state: GameState, context: EffectContext, node: ChoiceNode) -> int | None:
    value = get_effect_variable(state, context, f"{node.result_variable}{_HIDDEN_VALUE}")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise EffectInvariantError("serialized hidden-choice value stage is invalid")
    return value


def _hidden_zone_owner(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> PlayerId:
    """Return the player who disambiguates identities in the affected hidden zone."""

    if node.owner is not None:
        return resolve_player(node.owner, context, state)
    assert node.cards is not None
    if node.cards.player is not None:
        return resolve_player(node.cards.player, context, state)
    return resolve_player(node.chooser, context, state)


def _hidden_choice_is_direct(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> bool:
    """Whether the chooser may legally inspect the zone and so needs only one stage."""

    chooser = resolve_player(node.chooser, context, state)
    if chooser is _hidden_zone_owner(state, context, node, registry, programs):
        return True
    cards = _choice_cards(state, context, node, registry, programs)
    visible = _known_card_ids(observe(state, chooser, registry))
    return all(card_id in visible for card_id in cards)


def _choice_colors(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
) -> tuple[Color, ...]:
    target = node.target_player or node.chooser
    player = state.player(resolve_player(target, context, state))
    if node.color_source is ChoiceColorSource.PRESENT_ON_BOARD:
        # Decision 15: every colour the chooser currently has is legal, even when the resulting
        # splay is a no-op; an absent colour is not.
        colors = tuple(color for color in Color if player.board.stack(color).cards)
    elif node.color_source is ChoiceColorSource.PRESENT_IN_HAND:
        present = {registry.card(card_id).color for card_id in player.hand}
        colors = tuple(color for color in Color if color in present)
    else:
        colors = node.colors
    if node.required_splay is not None:
        colors = tuple(
            color for color in colors if player.board.stack(color).splay is node.required_splay
        )
    if node.minimum_stack_size:
        colors = tuple(
            color
            for color in colors
            if len(player.board.stack(color).cards) >= node.minimum_stack_size
        )
    return colors


def _choice_actions(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> tuple[SemanticAction, ...]:
    decision_id = state.next_decision_id
    actions: list[SemanticAction] = []
    if node.choice_kind is ChoiceKind.CARD:
        cards = _choice_cards(state, context, node, registry, programs)
        _require_exact_choice_visibility(state, context, node, cards, registry)
        actions.extend(ChooseCardAction(decision_id, card) for card in cards)
    elif node.choice_kind is ChoiceKind.BOUNDED_CARDS:
        selected = _selected_cards(state, context, node)
        # Decision 16: successive picks strictly increase by card ID, so one subset has exactly
        # one selection path and replay diffs stay meaningful. Mandatory partial execution lowers
        # the effective minimum only when the whole candidate set is genuinely too small; choices
        # that would strand an otherwise reachable minimum are never offered.
        candidates = _choice_cards(state, context, node, registry, programs)
        _require_exact_choice_visibility(state, context, node, candidates, registry)
        effective_minimum = min(node.minimum, len(candidates))
        floor = selected[-1].value if selected else ""
        available = tuple(
            card for card in candidates if card not in selected and card.value > floor
        )
        needed_after_pick = max(0, effective_minimum - len(selected) - 1)
        remaining = (
            ()
            if len(selected) >= node.maximum
            else tuple(
                card
                for card in available
                if sum(other.value > card.value for other in available) >= needed_after_pick
            )
        )
        actions.extend(ChooseCardAction(decision_id, card) for card in remaining)
        if len(selected) >= effective_minimum:
            actions.append(FinishSelectionAction(decision_id))
    elif node.choice_kind is ChoiceKind.HIDDEN_CARD:
        cards = _choice_cards(state, context, node, registry, programs)
        if _hidden_choice_is_direct(state, context, node, registry, programs):
            actions.extend(ChooseCardAction(decision_id, card) for card in cards)
        else:
            stage = _hidden_stage(state, context, node)
            if stage is None:
                actions.extend(
                    ChooseValueAction(decision_id, value)
                    for value in _hidden_values(state, context, node, registry, programs)
                )
            else:
                actions.extend(
                    ChooseCardAction(decision_id, card)
                    for card in cards
                    if registry.card(card).age == stage
                )
    elif node.choice_kind is ChoiceKind.COLOR:
        actions.extend(
            ChooseColorAction(decision_id, color)
            for color in _choice_colors(state, context, node, registry)
        )
    elif node.choice_kind is ChoiceKind.PLAYER:
        resolved = tuple(
            player_id
            for reference in node.players
            for player_id in resolve_players(reference, context, state)
        )
        players = tuple(dict.fromkeys(resolved))
        actions.extend(ChoosePlayerAction(decision_id, player) for player in players)
    elif node.choice_kind is ChoiceKind.VALUE:
        actions.extend(ChooseValueAction(decision_id, value) for value in node.values)
    elif node.choice_kind is ChoiceKind.SPLAY:
        actions.extend(ChooseSplayAction(decision_id, direction) for direction in node.directions)
    elif node.choice_kind is ChoiceKind.BRANCH:
        actions.extend(ChooseBranchAction(decision_id, branch) for branch in node.branches)
    else:
        cards = _choice_cards(state, context, node, registry, programs)
        _require_exact_choice_visibility(state, context, node, cards, registry)
        actions.extend(
            ChooseCardAction(decision_id, card)
            for card in _order_candidates(state, context, node, registry, programs)
        )
    if node.optional and node.choice_kind is not ChoiceKind.BOUNDED_CARDS:
        actions.append(DeclineAction(decision_id))
    return tuple(actions)


def _auto_choice(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> GameState | None:
    """Resolve a choice without a decision when no legal alternative exists.

    Returning a state means the choice is finished; returning ``None`` means a real decision must
    be raised. Automatic resolution is how partial execution stays decision-free.
    """

    actions = _choice_actions(state, context, node, registry, programs)
    substantive = tuple(action for action in actions if not isinstance(action, DeclineAction))
    if node.choice_kind is ChoiceKind.BOUNDED_CARDS:
        selected = _selected_cards(state, context, node)
        candidates = _choice_cards(state, context, node, registry, programs)
        effective_minimum = min(node.minimum, len(candidates))
        if len(selected) >= node.maximum:
            return state
        if not any(isinstance(action, ChooseCardAction) for action in actions):
            if len(selected) < effective_minimum:
                raise EffectInvariantError(
                    f"bounded choice {node.node_id} cannot reach its effective minimum"
                )
            return set_effect_variable(
                state,
                context,
                node.result_variable,
                tuple(card.value for card in selected),
            )
        return None
    if node.choice_kind is ChoiceKind.ORDER_CARDS:
        cards = _choice_cards(state, context, node, registry, programs)
        if not _order_needs_decision(cards, node, registry):
            return set_effect_variable(
                state,
                context,
                node.result_variable,
                tuple(card.value for card in _canonical_order(cards, node, registry)),
            )
        chosen = _order_progress(state, context, node)
        if len(chosen) >= len(cards):
            return state
        if not substantive:  # pragma: no cover - defensive
            return set_effect_variable(state, context, node.result_variable, None)
        if len(substantive) == 1 and isinstance(substantive[0], ChooseCardAction):
            forced = (*chosen, substantive[0].card_id)
            updated = set_effect_variable(
                state,
                context,
                f"{node.result_variable}{_ORDER_PROGRESS}",
                tuple(card.value for card in forced),
            )
            if len(forced) >= len(cards):
                return set_effect_variable(
                    updated,
                    context,
                    node.result_variable,
                    tuple(card.value for card in forced),
                )
            return _auto_choice(updated, context, node, registry, programs)
        return None
    if node.choice_kind is ChoiceKind.HIDDEN_CARD:
        cards = _choice_cards(state, context, node, registry, programs)
        if not cards:
            return set_effect_variable(state, context, node.result_variable, None)
        if len(cards) == 1:
            # Decision 13's stable fallback: exactly one candidate needs no disambiguation.
            return set_effect_variable(state, context, node.result_variable, cards[0].value)
        stage = _hidden_stage(state, context, node)
        if stage is not None:
            matching = tuple(card for card in cards if registry.card(card).age == stage)
            if len(matching) == 1:
                return set_effect_variable(state, context, node.result_variable, matching[0].value)
        return None
    if not substantive:
        return set_effect_variable(state, context, node.result_variable, None)
    return None


def _decision_context(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> DecisionContext:
    frozen = frozen_icon_counts(state)
    selected: tuple[CardId, ...] = ()
    minimum, maximum = node.minimum, node.maximum
    selection_kind = IncrementalSelectionKind.NONE
    if node.choice_kind is ChoiceKind.BOUNDED_CARDS:
        selected = _selected_cards(state, context, node)
        selection_kind = IncrementalSelectionKind.BOUNDED_SUBSET
    elif node.choice_kind is ChoiceKind.ORDER_CARDS:
        selected = _order_progress(state, context, node)
        total = len(_choice_cards(state, context, node, registry, programs))
        minimum = maximum = total
        selection_kind = IncrementalSelectionKind.CARD_ORDER
    return DecisionContext(
        demand=context.demand,
        shared=context.shared,
        nested=context.nested,
        featured_icon=frozen[0] if frozen is not None else None,
        activator_icons=frozen[1] if frozen is not None else None,
        opponent_icons=frozen[2] if frozen is not None else None,
        minimum_count=minimum,
        maximum_count=maximum,
        selected_so_far=selected,
        incremental_selection=selection_kind,
    )


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
    legal_actions = _choice_actions(state, context, node, registry, programs)
    if not legal_actions:
        return None
    chooser = _decision_chooser(state, context, node, registry, programs)
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
        _decision_context(state, context, node, registry, programs),
    )


def _decision_chooser(
    state: GameState,
    context: EffectContext,
    node: ChoiceNode,
    registry: CardRegistry,
    programs: EffectProgramRegistry | None = None,
) -> PlayerId:
    """Return who actually chooses, separating the effect executor from the chooser.

    Rules decision 13: for a hidden zone the executor makes the public semantic choice, then the
    zone owner disambiguates identities among equally public candidates.
    """

    if node.choice_kind is ChoiceKind.HIDDEN_CARD and not _hidden_choice_is_direct(
        state, context, node, registry, programs
    ):
        if _hidden_stage(state, context, node) is None:
            return resolve_player(node.chooser, context, state)
        return _hidden_zone_owner(state, context, node, registry, programs)
    return resolve_player(node.chooser, context, state)


def _advance_program(
    state: GameState,
    frame: EffectFrameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry,
) -> EffectResolution:
    program = programs.program(_program_id(frame))
    context = frame_context(frame)
    depth = _nested_depth(frame)
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
    completion_events: list[EffectEvent] = []
    if frame.step > 0:
        previous = entries[frame.step - 1]
        previous_context = context.for_effect(previous.effect_id, demand=previous.demand)
        # Decision 12 also requires a defensive printed-effect boundary. Every mutation leaf has
        # already checked, so this is normally a no-op; keeping it here prevents a malformed or
        # restored state from carrying latent eligibility into the next ordinal.
        baseline = qualifying_change_count(updated, context)
        updated, boundary_events, terminal = _achievement_boundary(
            updated, previous_context, registry
        )
        completion_events.extend(boundary_events)
        if terminal is not None:
            return EffectResolution(
                updated,
                EffectStatus.TERMINAL,
                events=tuple(completion_events),
                qualifying_changes=baseline + sum(event.changed for event in completion_events),
            )
        updated = clear_effect_scope(updated, previous_context)
        updated = clear_revealed_scope(updated, previous_context.scope)
    if frame.step >= len(entries):
        changes = qualifying_change_count(updated, context)
        updated = _pop(updated)
        if _is_root_program(frame):
            updated = clear_effect_scope(updated, context)
            updated = clear_revealed_scope(updated, context.scope)
            return EffectResolution(
                updated,
                EffectStatus.COMPLETE,
                events=tuple(completion_events),
                qualifying_changes=changes,
            )
        if context.nested:
            updated = clear_effect_scope(updated, context)
            updated = clear_revealed_scope(updated, context.scope)
        return EffectResolution(
            updated,
            EffectStatus.CONTINUE,
            events=tuple(completion_events),
        )
    effect = entries[frame.step]
    next_frame = replace(frame, step=frame.step + 1)
    updated = _replace_top(updated, next_frame)
    effect_context = context.for_effect(effect.effect_id, demand=effect.demand)
    updated = _push(
        updated,
        _node_frame(program.program_id, effect.root_node_id, effect_context, nested_depth=depth),
    )
    return EffectResolution(
        updated,
        EffectStatus.CONTINUE,
        events=tuple(completion_events),
    )


def _achievement_boundary(
    state: GameState,
    context: EffectContext,
    registry: CardRegistry,
    *,
    atomic_group_id: int | None = None,
) -> tuple[GameState, tuple[EffectEvent, ...], TerminalState | None]:
    """Run the WP6 achievement check at an atomic boundary.

    Decisions 3 and 12: predicates read live state, the active player is checked first, and a
    sixth achievement stops all remaining work including the sharing bonus.
    """

    result: AchievementCheckResult = check_atomic_boundary(
        state, registry, active_player=state.active_player
    )
    updated = result.state
    events: list[EffectEvent] = []
    for _claim in result.claims:
        updated, event = _event(
            updated,
            context,
            EffectEventKind.ACHIEVEMENT,
            achievement_player=_claim.player_id,
            achievement_id=_claim.achievement_id,
            atomic_group_id=atomic_group_id,
        )
        events.append(event)
    return updated, tuple(events), result.terminal


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
    depth = _nested_depth(frame)

    if isinstance(node, ChoiceNode):
        if frame.step == 0:
            state = delete_effect_variable(state, context, node.result_variable)
            state = delete_effect_variable(
                state, context, f"{node.result_variable}{_ORDER_PROGRESS}"
            )
            state = delete_effect_variable(state, context, f"{node.result_variable}{_HIDDEN_VALUE}")
            frame = replace(frame, step=1)
            state = _replace_top(state, frame)
        automatic = _auto_choice(state, context, node, registry, programs)
        if automatic is not None:
            return EffectResolution(_pop(automatic), EffectStatus.CONTINUE)
        decision = current_effect_decision(state, programs, registry)
        if decision is None:
            raise EffectInvariantError(f"choice node {node.node_id} produced no legal action")
        return EffectResolution(state, EffectStatus.AWAIT_DECISION, decision)
    if isinstance(node, CollectNode):
        card_id = _card_variable(state, context, node.card_variable)
        updated = state
        if card_id is not None:
            collected = _card_tuple_variable(state, context, node.result_variable)
            if card_id in collected:
                raise EffectInvariantError(
                    f"collection node {node.node_id} selected duplicate card {card_id}"
                )
            updated = set_effect_variable(
                state,
                context,
                node.result_variable,
                tuple(item.value for item in (*collected, card_id)),
            )
        return EffectResolution(_pop(updated), EffectStatus.CONTINUE)
    if isinstance(node, SequenceNode):
        if frame.step >= len(node.children):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = _push(
            updated, _node_frame(program_id, node.children[frame.step], context, nested_depth=depth)
        )
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, ConditionNode):
        branch = (
            node.when_true
            if evaluate_predicate(state, context, node.predicate, registry, programs)
            else node.when_false
        )
        updated = _pop(state)
        if branch is not None:
            updated = _push(updated, _node_frame(program_id, branch, context, nested_depth=depth))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, AllOrNoneNode):
        # The guard runs before the body so a partial prefix can never be performed.
        if not evaluate_predicate(state, context, node.guard, registry, programs):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        updated = _pop(state)
        updated = _push(updated, _node_frame(program_id, node.body, context, nested_depth=depth))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, RepeatNode):
        if frame.step == 0:
            updated = _replace_top(state, replace(frame, step=1))
            updated = _push(
                updated, _node_frame(program_id, node.body, context, nested_depth=depth)
            )
            return EffectResolution(updated, EffectStatus.CONTINUE)
        if not evaluate_predicate(state, context, node.repeat_if, registry, programs):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        if frame.step >= node.maximum_iterations:
            raise EffectInvariantError(
                f"repeat node {node.node_id} exceeded {node.maximum_iterations} iterations"
            )
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = _push(updated, _node_frame(program_id, node.body, context, nested_depth=depth))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, TimesNode):
        # Decision 17: the count is resolved once, on entry, and stored in the frame.
        if frame.step == 0:
            total = max(0, resolve_value(state, context, node.count, registry, programs))
            if total > node.maximum_iterations:
                raise EffectInvariantError(
                    f"times node {node.node_id} requested {total} iterations, "
                    f"exceeding {node.maximum_iterations}"
                )
            updated = set_effect_variable(state, context, f"{node.node_id}{_TIMES_LIMIT}", total)
            if total == 0:
                return EffectResolution(_pop(updated), EffectStatus.CONTINUE)
            updated = _replace_top(updated, replace(frame, step=1))
            if node.index_variable is not None:
                updated = set_effect_variable(updated, context, node.index_variable, 1)
            updated = _push(
                updated, _node_frame(program_id, node.body, context, nested_depth=depth)
            )
            return EffectResolution(updated, EffectStatus.CONTINUE)
        raw_total = get_effect_variable(state, context, f"{node.node_id}{_TIMES_LIMIT}", 0)
        if not isinstance(raw_total, int) or isinstance(raw_total, bool):
            raise EffectInvariantError("serialized times-node count is invalid")
        if frame.step >= raw_total:
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        if node.index_variable is not None:
            updated = set_effect_variable(updated, context, node.index_variable, frame.step + 1)
        updated = _push(updated, _node_frame(program_id, node.body, context, nested_depth=depth))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, ForEachCardNode):
        cards = _card_tuple_variable(state, context, node.cards_variable)
        if len(cards) > node.maximum_iterations:
            raise EffectInvariantError(
                f"for-each node {node.node_id} iterates {len(cards)} cards, "
                f"exceeding {node.maximum_iterations}"
            )
        if frame.step >= len(cards):
            return EffectResolution(_pop(state), EffectStatus.CONTINUE)
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = set_effect_variable(updated, context, node.item_variable, cards[frame.step].value)
        updated = _push(updated, _node_frame(program_id, node.body, context, nested_depth=depth))
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, BatchNode):
        updated = state
        all_events: list[EffectEvent] = []
        atomic_group_id = state.next_event_id
        batch_baseline = qualifying_change_count(updated, context)
        for child_id in node.children:
            child = program.node(child_id)
            updated, events, status = _execute_leaf(
                updated,
                context,
                child,
                registry,
                programs,
                atomic_group_id=atomic_group_id,
            )
            all_events.extend(events)
            if status is EffectStatus.TERMINAL:
                changes = batch_baseline + sum(event.changed for event in all_events)
                return EffectResolution(
                    updated,
                    status,
                    events=tuple(all_events),
                    qualifying_changes=changes,
                )
        # Decision 4/12: a batch exposes no intermediate decision, so its achievement check runs
        # exactly once, on exit.
        updated, boundary_events, terminal = _achievement_boundary(
            updated,
            context,
            registry,
            atomic_group_id=atomic_group_id,
        )
        all_events.extend(boundary_events)
        if terminal is not None:
            changes = batch_baseline + sum(event.changed for event in all_events)
            return EffectResolution(
                updated,
                EffectStatus.TERMINAL,
                events=tuple(all_events),
                qualifying_changes=changes,
            )
        return EffectResolution(_pop(updated), EffectStatus.CONTINUE, events=tuple(all_events))
    if isinstance(node, NestedNode):
        card_id = _card_variable(state, context, node.card_variable)
        updated = _pop(state)
        if card_id is None:
            return EffectResolution(updated, EffectStatus.CONTINUE)
        nested_program = programs.program_for_card(card_id)
        root = _root_context(context)
        count = get_effect_variable(state, root, _NESTED_COUNT, 0)
        if not isinstance(count, int) or isinstance(count, bool):
            raise EffectInvariantError("serialized nested execution count is invalid")
        # Decision 6: nesting depth is its own serialized budget, not a step count, so
        # Robotics -> Software -> Robotics fails loudly instead of silently truncating.
        if depth + 1 > MAX_NESTED_DEPTH:
            raise EffectInvariantError(
                f"nested execution depth {depth + 1} exceeded the limit {MAX_NESTED_DEPTH} "
                f"in dogma action {context.dogma_action_id}"
            )
        updated = set_effect_variable(updated, root, _NESTED_COUNT, count + 1)
        nested_context = context.for_nested(card_id, f"nested-{count + 1}")
        updated = _push(
            updated,
            _program_frame(
                nested_program,
                nested_context,
                non_demand_only=True,
                nested_depth=depth + 1,
            ),
        )
        return EffectResolution(updated, EffectStatus.CONTINUE)
    if isinstance(node, StopNode):
        # End this printed effect only: drop node frames up to and including this effect's root.
        return EffectResolution(_stop_effect(state), EffectStatus.CONTINUE)
    if isinstance(node, AbortDogmaNode):
        changes = qualifying_change_count(state, context)
        updated, event = _event(state, context, EffectEventKind.ABORT_DOGMA)
        updated = replace(updated, pending_effects=(), effect_variables=(), revealed=())
        return EffectResolution(
            updated,
            EffectStatus.ABORT_DOGMA,
            events=(event,),
            qualifying_changes=changes,
        )

    leaf_group_id = state.next_event_id if isinstance(node, DrawAndMoveNode) else None
    leaf_baseline = qualifying_change_count(state, context)
    updated, events, status = _execute_leaf(
        state,
        context,
        node,
        registry,
        programs,
        atomic_group_id=leaf_group_id,
    )
    if status is EffectStatus.TERMINAL:
        changes = leaf_baseline + sum(event.changed for event in events)
        return EffectResolution(
            updated,
            status,
            events=events,
            qualifying_changes=changes,
        )
    all_leaf_events = list(events)
    # Decision 12: one card-effect instruction is the normal atomic boundary, so a non-batch leaf
    # runs the achievement check immediately after its mutation.
    updated, boundary_events, terminal = _achievement_boundary(
        updated,
        context,
        registry,
        atomic_group_id=leaf_group_id,
    )
    all_leaf_events.extend(boundary_events)
    if terminal is not None:
        changes = leaf_baseline + sum(event.changed for event in all_leaf_events)
        return EffectResolution(
            updated,
            EffectStatus.TERMINAL,
            events=tuple(all_leaf_events),
            qualifying_changes=changes,
        )
    return EffectResolution(_pop(updated), EffectStatus.CONTINUE, events=tuple(all_leaf_events))


def _stop_effect(state: GameState) -> GameState:
    """Pop node frames until the enclosing program frame is exposed.

    ``Stop`` ends only the current printed effect: the program frame below decides whether another
    printed ordinal, another executor, or the sharing bonus still has work.
    """

    frames = list(state.pending_effects)
    while frames and frames[-1].kind == _NODE_FRAME:
        frames.pop()
    return replace(state, pending_effects=tuple(frames))


def step_effect(
    state: GameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> EffectResolution:
    """Advance exactly one explicit frame operation, suitable for checkpointing."""

    registry = registry or load_card_registry()
    if state.phase is GamePhase.TERMINAL:
        return EffectResolution(state, EffectStatus.TERMINAL)
    if not state.pending_effects:
        return EffectResolution(state, EffectStatus.COMPLETE)
    frame = state.pending_effects[-1]
    if frame.kind == DOGMA_FRAME:
        # WP5's orchestrator owns this frame kind. The import is local so the node interpreter
        # and the dogma step machine can call into each other without an import cycle.
        from .dogma import step_dogma

        return step_dogma(state, frame, programs, registry)
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
        return _advance_program(state, frame, programs, registry)
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
    if state.pending_effects or state.effect_variables:
        raise EffectInvariantError("cannot start a root effect while effect runtime is pending")
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
            selected = (*_selected_cards(updated, context, node), action.card_id)
            updated = set_effect_variable(
                updated,
                context,
                node.result_variable,
                tuple(card.value for card in selected),
            )
            complete = len(selected) >= node.maximum
        elif node.choice_kind is ChoiceKind.ORDER_CARDS:
            chosen = (*_order_progress(updated, context, node), action.card_id)
            updated = set_effect_variable(
                updated,
                context,
                f"{node.result_variable}{_ORDER_PROGRESS}",
                tuple(card.value for card in chosen),
            )
            total = len(_choice_cards(updated, context, node, registry, programs))
            complete = len(chosen) >= total
            if complete:
                updated = set_effect_variable(
                    updated,
                    context,
                    node.result_variable,
                    tuple(card.value for card in chosen),
                )
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
        if node.choice_kind is ChoiceKind.HIDDEN_CARD:
            # Stage one of decision 13: the executor fixes the public projection, then the zone
            # owner disambiguates identities without seeing the alternatives.
            updated = set_effect_variable(
                updated,
                context,
                f"{node.result_variable}{_HIDDEN_VALUE}",
                action.value,
            )
            complete = False
        else:
            updated = set_effect_variable(updated, context, node.result_variable, action.value)
    elif isinstance(action, ChooseSplayAction):
        updated = set_effect_variable(
            updated, context, node.result_variable, action.direction.value
        )
    elif isinstance(action, ChooseBranchAction):
        updated = set_effect_variable(updated, context, node.result_variable, action.branch_id)
    else:  # pragma: no cover - legal effect actions exhaust the choice kinds
        raise EffectInvariantError(f"unsupported effect choice action: {action.kind}")
    if complete:
        updated = _pop(updated)
    updated = replace(updated, next_decision_id=updated.next_decision_id + 1)
    return resume_effect(updated, programs, registry)
