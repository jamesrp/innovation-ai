"""Declarative effect programs interpreted by the resumable VM."""

from __future__ import annotations

import hashlib
import inspect
import json
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

# ``ZoneKind`` is re-exported here so a card module can name a movement destination without
# importing the mutation layer.
from innovation_ai.innovation.zones import ZoneKind as ZoneKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from innovation_ai.innovation.catalog import CardRegistry
    from innovation_ai.innovation.state import GameState

    from .model import EffectContext


class NamedPredicate(Protocol):
    """Bounded escape hatch for a card's genuinely singular boolean test.

    Implementations must be pure: they read state and context and return a boolean. They never
    mutate state, so no card gets a private mutation path.
    """

    def __call__(
        self,
        state: GameState,
        context: EffectContext,
        registry: CardRegistry,
    ) -> bool: ...


class NamedValue(Protocol):
    """Bounded escape hatch for a card's genuinely singular integer quantity."""

    def __call__(
        self,
        state: GameState,
        context: EffectContext,
        registry: CardRegistry,
    ) -> int: ...


class PlayerRefKind(StrEnum):
    """Ways an effect can identify a player without a callback."""

    ACTOR = "actor"
    CHOOSER = "chooser"
    EXECUTOR = "executor"
    ACTIVATOR = "activator"
    OPPONENT_OF_EXECUTOR = "opponent-of-executor"
    ALL = "all"
    ALL_OTHER = "all-other"
    LITERAL = "literal"
    VARIABLE = "variable"


_MULTI_PLAYER_KINDS = frozenset({PlayerRefKind.ALL, PlayerRefKind.ALL_OTHER})


@dataclass(frozen=True, slots=True)
class PlayerRef:
    """Serializable player reference used by choices and movements.

    ``ALL``/``ALL_OTHER`` denote a *set* of players in canonical order. They are legal only where
    a set makes sense - card selectors and player choices - and raise when a single destination or
    executor is required, so "any player's board" cannot silently become one player's board.
    """

    kind: PlayerRefKind
    player_id: PlayerId | None = None
    variable: str | None = None

    def __post_init__(self) -> None:
        if (self.player_id is not None) != (self.kind is PlayerRefKind.LITERAL):
            raise ValueError("only a literal player reference carries a player ID")
        if (self.variable is not None) != (self.kind is PlayerRefKind.VARIABLE):
            raise ValueError("only a variable player reference carries a variable name")

    @property
    def is_multi(self) -> bool:
        """Whether this reference denotes a set of players rather than one."""

        return self.kind in _MULTI_PLAYER_KINDS

    @classmethod
    def literal(cls, player_id: PlayerId) -> PlayerRef:
        return cls(PlayerRefKind.LITERAL, player_id)

    @classmethod
    def from_variable(cls, variable: str) -> PlayerRef:
        return cls(PlayerRefKind.VARIABLE, variable=variable)


EXECUTOR = PlayerRef(PlayerRefKind.EXECUTOR)
ACTIVATOR = PlayerRef(PlayerRefKind.ACTIVATOR)
CHOOSER = PlayerRef(PlayerRefKind.CHOOSER)
OPPONENT = PlayerRef(PlayerRefKind.OPPONENT_OF_EXECUTOR)
ALL_PLAYERS = PlayerRef(PlayerRefKind.ALL)
ALL_OTHER_PLAYERS = PlayerRef(PlayerRefKind.ALL_OTHER)


class CardSelectorKind(StrEnum):
    """Supported deterministic card-set queries."""

    HAND = "hand"
    SCORE = "score"
    BOARD_STACK = "board-stack"
    BOARD_ALL = "board-all"
    TOP_CARDS = "top-cards"
    VARIABLE = "variable"
    CONSTANT = "constant"


class Cmp(StrEnum):
    """Comparators available to value filters and count conditions."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class Extreme(StrEnum):
    """Which end of a value range an effect selects."""

    HIGHEST = "highest"
    LOWEST = "lowest"


class ExtremeScope(StrEnum):
    """Whether tied extremes all qualify or the chooser picks one.

    ``ALL_TIED`` matches "all the highest cards"; ``ONE_TIED`` matches "the highest card", where
    the rules keyword section makes the owner/controller of the choice break the tie.
    """

    ALL_TIED = "all-tied"
    ONE_TIED = "one-tied"


class StackPosition(StrEnum):
    """Positional restriction inside one color stack."""

    ANY = "any"
    TOP = "top"
    BOTTOM = "bottom"
    BENEATH_TOP = "beneath-top"
    BENEATH_VARIABLE = "beneath-variable"


class SelectorRelationKind(StrEnum):
    """Relational colour/value tests against a second card set."""

    SAME_COLOR_AS_ANY = "same-color-as-any"
    DIFFERENT_COLOR_FROM_ALL = "different-color-from-all"
    SAME_VALUE_AS_ANY = "same-value-as-any"


@dataclass(frozen=True, slots=True)
class SelectorRelation:
    """Restrict a candidate set by a relation to another declarative card set."""

    kind: SelectorRelationKind
    reference: CardSelector


@dataclass(frozen=True, slots=True)
class CardSelector:
    """A declarative card query evaluated against live authoritative state.

    Filters compose in a fixed order so a selector always has one meaning: zone/position, colour,
    icon, value, relation, named predicate, source-card and variable exclusions, and finally the
    extreme filter. ``predicate`` is the bounded escape hatch resolved from the owning card
    module's ``PREDICATES`` mapping; it never gets its own mutation path.
    """

    kind: CardSelectorKind
    player: PlayerRef | None = None
    color: Color | None = None
    color_variable: str | None = None
    variable: str | None = None
    cards: tuple[CardId, ...] = ()
    icon: Icon | None = None
    without_icon: Icon | None = None
    colors: tuple[Color, ...] = ()
    exclude_colors: tuple[Color, ...] = ()
    value: int | None = None
    value_expr: ValueRef | None = None
    value_cmp: Cmp = Cmp.EQ
    value_offset: int = 0
    position: StackPosition = StackPosition.ANY
    position_variable: str | None = None
    extreme: Extreme | None = None
    extreme_scope: ExtremeScope = ExtremeScope.ALL_TIED
    exclude_variable: str | None = None
    relation: SelectorRelation | None = None
    predicate: str | None = None
    exclude_source_card: bool = False

    def __post_init__(self) -> None:
        player_kinds = {
            CardSelectorKind.HAND,
            CardSelectorKind.SCORE,
            CardSelectorKind.BOARD_STACK,
            CardSelectorKind.BOARD_ALL,
            CardSelectorKind.TOP_CARDS,
        }
        if (self.player is not None) != (self.kind in player_kinds):
            raise ValueError(f"{self.kind} has an invalid player reference")
        if self.kind is CardSelectorKind.BOARD_STACK:
            if (self.color is None) == (self.color_variable is None):
                raise ValueError("a board-stack selector needs one color source")
        elif self.color is not None or self.color_variable is not None:
            raise ValueError("only a board-stack selector carries a color")
        if (self.variable is not None) != (self.kind is CardSelectorKind.VARIABLE):
            raise ValueError("only a variable selector carries a variable name")
        if bool(self.cards) != (self.kind is CardSelectorKind.CONSTANT):
            raise ValueError("only a non-empty constant selector carries card IDs")
        if self.value is not None and self.value_expr is not None:
            raise ValueError("a value filter has one literal or expression source")
        if self.position is not StackPosition.ANY and self.kind not in {
            CardSelectorKind.BOARD_STACK,
            CardSelectorKind.BOARD_ALL,
        }:
            raise ValueError("stack positions apply only to board selectors")
        if (self.position_variable is not None) != (
            self.position is StackPosition.BENEATH_VARIABLE
        ):
            raise ValueError("only a beneath-variable position carries a variable name")
        if set(self.colors) & set(self.exclude_colors):
            raise ValueError("a colour cannot be both required and excluded")
        if len(set(self.colors)) != len(self.colors) or len(set(self.exclude_colors)) != len(
            self.exclude_colors
        ):
            raise ValueError("selector colour filters cannot repeat a colour")

    @classmethod
    def hand(
        cls,
        player: PlayerRef = EXECUTOR,
        *,
        icon: Icon | None = None,
        extreme: Extreme | None = None,
        extreme_scope: ExtremeScope = ExtremeScope.ALL_TIED,
        value: int | None = None,
        value_cmp: Cmp = Cmp.EQ,
    ) -> CardSelector:
        return cls(
            CardSelectorKind.HAND,
            player,
            icon=icon,
            extreme=extreme,
            extreme_scope=extreme_scope,
            value=value,
            value_cmp=value_cmp,
        )

    @classmethod
    def score(
        cls,
        player: PlayerRef = EXECUTOR,
        *,
        extreme: Extreme | None = None,
        extreme_scope: ExtremeScope = ExtremeScope.ALL_TIED,
        value: int | None = None,
        value_cmp: Cmp = Cmp.EQ,
    ) -> CardSelector:
        return cls(
            CardSelectorKind.SCORE,
            player,
            extreme=extreme,
            extreme_scope=extreme_scope,
            value=value,
            value_cmp=value_cmp,
        )

    @classmethod
    def top_cards(
        cls,
        player: PlayerRef = EXECUTOR,
        *,
        exclude_source_card: bool = False,
        icon: Icon | None = None,
        without_icon: Icon | None = None,
        colors: tuple[Color, ...] = (),
        exclude_colors: tuple[Color, ...] = (),
        extreme: Extreme | None = None,
        extreme_scope: ExtremeScope = ExtremeScope.ALL_TIED,
    ) -> CardSelector:
        return cls(
            CardSelectorKind.TOP_CARDS,
            player,
            exclude_source_card=exclude_source_card,
            icon=icon,
            without_icon=without_icon,
            colors=colors,
            exclude_colors=exclude_colors,
            extreme=extreme,
            extreme_scope=extreme_scope,
        )

    @classmethod
    def board(cls, player: PlayerRef = EXECUTOR, **kwargs: object) -> CardSelector:
        return cls(CardSelectorKind.BOARD_ALL, player, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def stack(
        cls,
        player: PlayerRef = EXECUTOR,
        *,
        color: Color | None = None,
        color_variable: str | None = None,
        position: StackPosition = StackPosition.ANY,
        position_variable: str | None = None,
    ) -> CardSelector:
        return cls(
            CardSelectorKind.BOARD_STACK,
            player,
            color=color,
            color_variable=color_variable,
            position=position,
            position_variable=position_variable,
        )

    @classmethod
    def from_variable(cls, variable: str) -> CardSelector:
        return cls(CardSelectorKind.VARIABLE, variable=variable)

    @classmethod
    def constant(cls, cards: tuple[CardId, ...]) -> CardSelector:
        return cls(CardSelectorKind.CONSTANT, cards=cards)


class Rounding(StrEnum):
    """How a divided quantity rounds."""

    FLOOR = "floor"
    CEIL = "ceil"


class ValueRefKind(StrEnum):
    """Supported integer expression forms."""

    LITERAL = "literal"
    VARIABLE = "variable"
    COUNT_CARDS = "count-cards"
    COUNT_SELECTOR = "count-selector"
    ICON_COUNT = "icon-count"
    COLORS_WITH_ICON = "colors-with-icon"
    COLORS_PRESENT_ONLY_HERE = "colors-present-only-here"
    COLORS_SPLAYED = "colors-splayed"
    DISTINCT_VALUES = "distinct-values"
    CARD_VALUE = "card-value"
    SELECTOR_EXTREME = "selector-extreme"
    SCORE = "score"
    ACHIEVEMENT_COUNT = "achievement-count"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class ValueRef:
    """A small serializable integer expression.

    Evaluation timing is fixed by rules decision 17: a quantity is evaluated once when the
    instruction that owns it begins. ``Repeat`` and an explicit repeat of a dogma effect start a
    new execution and therefore reevaluate.

    The result is ``(raw // per`` or ``ceil(raw / per)) + offset``, floored at zero for counts.
    """

    kind: ValueRefKind
    value: int | None = None
    variable: str | None = None
    selector: CardSelector | None = None
    icon: Icon | None = None
    player: PlayerRef | None = None
    direction: SplayDirection | None = None
    extreme: Extreme | None = None
    name: str | None = None
    per: int = 1
    offset: int = 0
    rounding: Rounding = Rounding.FLOOR

    def __post_init__(self) -> None:
        if (self.value is not None) != (self.kind is ValueRefKind.LITERAL):
            raise ValueError("only a literal value reference carries a value")
        variable_kinds = {
            ValueRefKind.VARIABLE,
            ValueRefKind.COUNT_CARDS,
            ValueRefKind.DISTINCT_VALUES,
            ValueRefKind.CARD_VALUE,
        }
        if (self.variable is not None) != (self.kind in variable_kinds):
            raise ValueError(f"{self.kind} has an invalid variable reference")
        selector_kinds = {ValueRefKind.COUNT_SELECTOR, ValueRefKind.SELECTOR_EXTREME}
        if (self.selector is not None) != (self.kind in selector_kinds):
            raise ValueError(f"{self.kind} has an invalid card selector")
        icon_kinds = {ValueRefKind.ICON_COUNT, ValueRefKind.COLORS_WITH_ICON}
        if (self.icon is not None) != (self.kind in icon_kinds):
            raise ValueError(f"{self.kind} has an invalid icon")
        if (self.direction is not None) != (self.kind is ValueRefKind.COLORS_SPLAYED):
            raise ValueError("only a splay-count reference carries a direction")
        if (self.extreme is not None) != (self.kind is ValueRefKind.SELECTOR_EXTREME):
            raise ValueError("only a selector-extreme reference carries an extreme")
        if (self.name is not None) != (self.kind is ValueRefKind.NAMED):
            raise ValueError("only a named reference carries a name")
        player_kinds = {
            ValueRefKind.ICON_COUNT,
            ValueRefKind.COLORS_WITH_ICON,
            ValueRefKind.COLORS_PRESENT_ONLY_HERE,
            ValueRefKind.COLORS_SPLAYED,
            ValueRefKind.SCORE,
            ValueRefKind.ACHIEVEMENT_COUNT,
        }
        if self.kind in player_kinds and self.player is None:
            raise ValueError(f"{self.kind} needs an explicit player reference")
        if self.per < 1:
            raise ValueError("a quantity divisor must be positive")

    @classmethod
    def literal(cls, value: int) -> ValueRef:
        return cls(ValueRefKind.LITERAL, value=value)

    @classmethod
    def from_variable(cls, variable: str) -> ValueRef:
        return cls(ValueRefKind.VARIABLE, variable=variable)

    @classmethod
    def count(cls, variable: str) -> ValueRef:
        return cls(ValueRefKind.COUNT_CARDS, variable=variable)

    @classmethod
    def count_selector(cls, selector: CardSelector, *, per: int = 1, offset: int = 0) -> ValueRef:
        return cls(ValueRefKind.COUNT_SELECTOR, selector=selector, per=per, offset=offset)

    @classmethod
    def icon_count(
        cls, icon: Icon, player: PlayerRef = EXECUTOR, *, per: int = 1, offset: int = 0
    ) -> ValueRef:
        return cls(ValueRefKind.ICON_COUNT, icon=icon, player=player, per=per, offset=offset)

    @classmethod
    def card_value(cls, variable: str, *, offset: int = 0) -> ValueRef:
        return cls(ValueRefKind.CARD_VALUE, variable=variable, offset=offset)

    @classmethod
    def selector_extreme(
        cls, selector: CardSelector, extreme: Extreme = Extreme.HIGHEST, *, offset: int = 0
    ) -> ValueRef:
        return cls(
            ValueRefKind.SELECTOR_EXTREME,
            selector=selector,
            extreme=extreme,
            offset=offset,
        )

    @classmethod
    def named(cls, name: str) -> ValueRef:
        return cls(ValueRefKind.NAMED, name=name)


class VariableScope(StrEnum):
    """Scope used for values that must survive one printed-effect boundary."""

    LOCAL = "local"
    ROOT = "root"


class PredicateKind(StrEnum):
    """Supported conditions for branches, repeats, and guards."""

    VARIABLE_TRUTHY = "variable-truthy"
    VARIABLE_EQUALS = "variable-equals"
    CARD_HAS_ICON = "card-has-icon"
    CARD_COLOR_IS = "card-color-is"
    CARD_COLOR_IN = "card-color-in"
    COUNT_CMP = "count-cmp"
    ALL_CARDS_MATCH = "all-cards-match"
    ANY_CARDS_MATCH = "any-cards-match"
    SELECTOR_NON_EMPTY = "selector-non-empty"
    NAMED = "named"
    NOT = "not"


@dataclass(frozen=True, slots=True)
class Predicate:
    """A deterministic condition over scoped variables, state, and catalog facts.

    ``ALL_CARDS_MATCH`` is true for an empty candidate set, implementing rules decision 10 once
    for every universal card test.
    """

    kind: PredicateKind
    variable: str | None = None
    icon: Icon | None = None
    color: Color | None = None
    colors: tuple[Color, ...] = ()
    colors_variable: str | None = None
    expected: str | int | bool | None = None
    left: ValueRef | None = None
    right: ValueRef | None = None
    comparator: Cmp = Cmp.GE
    cards: CardSelector | None = None
    match: CardSelector | None = None
    name: str | None = None
    operand: Predicate | None = None
    variable_scope: VariableScope = VariableScope.LOCAL

    def __post_init__(self) -> None:
        variable_kinds = {
            PredicateKind.VARIABLE_TRUTHY,
            PredicateKind.VARIABLE_EQUALS,
            PredicateKind.CARD_HAS_ICON,
            PredicateKind.CARD_COLOR_IS,
            PredicateKind.CARD_COLOR_IN,
        }
        if (self.variable is not None) != (self.kind in variable_kinds):
            raise ValueError(f"{self.kind} has an invalid variable reference")
        if self.variable is not None and not self.variable:
            raise ValueError("predicate variable cannot be empty")
        if (self.icon is not None) != (self.kind is PredicateKind.CARD_HAS_ICON):
            raise ValueError("only an icon predicate carries an icon")
        if (self.color is not None) != (self.kind is PredicateKind.CARD_COLOR_IS):
            raise ValueError("only a colour predicate carries a colour")
        if self.kind is PredicateKind.CARD_COLOR_IN:
            if bool(self.colors) == (self.colors_variable is not None):
                raise ValueError("a colour-set predicate needs exactly one colour source")
        elif self.colors or self.colors_variable is not None:
            raise ValueError("only a colour-set predicate carries a colour set")
        if (self.expected is not None) != (self.kind is PredicateKind.VARIABLE_EQUALS):
            raise ValueError("only an equality predicate carries an expected value")
        if ((self.left is not None) or (self.right is not None)) != (
            self.kind is PredicateKind.COUNT_CMP
        ):
            raise ValueError("only a count comparison carries value operands")
        if self.kind is PredicateKind.COUNT_CMP and (self.left is None or self.right is None):
            raise ValueError("a count comparison needs both operands")
        selector_kinds = {
            PredicateKind.ALL_CARDS_MATCH,
            PredicateKind.ANY_CARDS_MATCH,
            PredicateKind.SELECTOR_NON_EMPTY,
        }
        if (self.cards is not None) != (self.kind in selector_kinds):
            raise ValueError(f"{self.kind} has an invalid card selector")
        if self.match is not None and self.kind not in {
            PredicateKind.ALL_CARDS_MATCH,
            PredicateKind.ANY_CARDS_MATCH,
        }:
            raise ValueError("only a universal card test carries a matching selector")
        if (self.name is not None) != (self.kind is PredicateKind.NAMED):
            raise ValueError("only a named predicate carries a name")
        if (self.operand is not None) != (self.kind is PredicateKind.NOT):
            raise ValueError("only a negation carries an operand")

    @classmethod
    def truthy(cls, variable: str, *, scope: VariableScope = VariableScope.LOCAL) -> Predicate:
        return cls(PredicateKind.VARIABLE_TRUTHY, variable, variable_scope=scope)

    @classmethod
    def equals(cls, variable: str, expected: str | int | bool) -> Predicate:
        return cls(PredicateKind.VARIABLE_EQUALS, variable, expected=expected)

    @classmethod
    def card_has_icon(cls, variable: str, icon: Icon) -> Predicate:
        return cls(PredicateKind.CARD_HAS_ICON, variable, icon=icon)

    @classmethod
    def card_color_is(cls, variable: str, color: Color) -> Predicate:
        return cls(PredicateKind.CARD_COLOR_IS, variable, color=color)

    @classmethod
    def card_color_in(
        cls,
        variable: str,
        colors: tuple[Color, ...] = (),
        *,
        colors_variable: str | None = None,
    ) -> Predicate:
        return cls(
            PredicateKind.CARD_COLOR_IN,
            variable,
            colors=colors,
            colors_variable=colors_variable,
        )

    @classmethod
    def count(cls, left: ValueRef, comparator: Cmp, right: ValueRef) -> Predicate:
        return cls(PredicateKind.COUNT_CMP, left=left, comparator=comparator, right=right)

    @classmethod
    def all_match(cls, cards: CardSelector, match: CardSelector) -> Predicate:
        return cls(PredicateKind.ALL_CARDS_MATCH, cards=cards, match=match)

    @classmethod
    def any_match(cls, cards: CardSelector, match: CardSelector) -> Predicate:
        return cls(PredicateKind.ANY_CARDS_MATCH, cards=cards, match=match)

    @classmethod
    def non_empty(cls, cards: CardSelector) -> Predicate:
        return cls(PredicateKind.SELECTOR_NON_EMPTY, cards=cards)

    @classmethod
    def named(cls, name: str) -> Predicate:
        return cls(PredicateKind.NAMED, name=name)

    @classmethod
    def negate(cls, operand: Predicate) -> Predicate:
        return cls(PredicateKind.NOT, operand=operand)


class OrderGroup(StrEnum):
    """Which movements an order decision can actually distinguish.

    Rules decision 5/16: ask for an order only when it can change authoritative state. Cards
    returned to different age piles, or melded/tucked into different colour stacks, cannot be
    distinguished by order, so those groups are ordered canonically without a decision.
    """

    ALL = "all"
    AGE = "age"
    COLOR = "color"


class ChoiceColorSource(StrEnum):
    """Where a colour choice's legal options come from.

    Rules decision 15 makes "you may splay X" offer every colour the chooser currently has on
    their board, even when the resulting splay is a no-op. ``PRESENT_ON_BOARD`` expresses that
    without each card restating the colour list.
    """

    EXPLICIT = "explicit"
    PRESENT_ON_BOARD = "present-on-board"


class ChoiceKind(StrEnum):
    """Effect choice shapes mapped to WP3 semantic actions."""

    CARD = "card"
    BOUNDED_CARDS = "bounded-cards"
    HIDDEN_CARD = "hidden-card"
    COLOR = "color"
    PLAYER = "player"
    VALUE = "value"
    SPLAY = "splay"
    BRANCH = "branch"
    ORDER_CARDS = "order-cards"


@dataclass(frozen=True, slots=True)
class ChoiceNode:
    """Pause for a deterministic first-class effect choice.

    ``ORDER_CARDS`` is resolved incrementally: the chooser repeatedly picks the next card, so an
    order over ``k`` cards costs ``k`` decisions of at most ``k`` actions instead of ``k!``
    enumerated permutations. ``BOUNDED_CARDS`` is likewise incremental and, per rules decision 16,
    canonicalised so one subset has exactly one selection path.

    ``HIDDEN_CARD`` implements rules decision 13's two-stage choice: the executor first chooses the
    public projection (a value), then the zone owner disambiguates identities. When the executor
    already owns or may inspect the zone, both stages collapse into the owner's single choice.
    """

    node_id: str
    choice_kind: ChoiceKind
    result_variable: str
    chooser: PlayerRef = CHOOSER
    target_player: PlayerRef | None = None
    cards: CardSelector | None = None
    colors: tuple[Color, ...] = ()
    players: tuple[PlayerRef, ...] = ()
    values: tuple[int, ...] = ()
    directions: tuple[SplayDirection, ...] = ()
    branches: tuple[str, ...] = ()
    minimum: int = 1
    maximum: int = 1
    optional: bool = False
    order_group: OrderGroup = OrderGroup.ALL
    minimum_stack_size: int = 0
    owner: PlayerRef | None = None
    color_source: ChoiceColorSource = ChoiceColorSource.EXPLICIT
    required_splay: SplayDirection | None = None

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("invalid choice bounds")
        card_kinds = {
            ChoiceKind.CARD,
            ChoiceKind.BOUNDED_CARDS,
            ChoiceKind.ORDER_CARDS,
            ChoiceKind.HIDDEN_CARD,
        }
        if (self.cards is not None) != (self.choice_kind in card_kinds):
            raise ValueError("card-shaped choices require exactly one card selector")
        if self.choice_kind is ChoiceKind.COLOR:
            if bool(self.colors) != (self.color_source is ChoiceColorSource.EXPLICIT):
                raise ValueError("a colour choice needs exactly one option source")
        elif self.color_source is not ChoiceColorSource.EXPLICIT:
            raise ValueError("only a colour choice derives its options from the board")
        if self.required_splay is not None and self.choice_kind is not ChoiceKind.COLOR:
            raise ValueError("only a colour choice filters by current splay")
        options = {
            ChoiceKind.PLAYER: bool(self.players),
            ChoiceKind.VALUE: bool(self.values),
            ChoiceKind.SPLAY: bool(self.directions),
            ChoiceKind.BRANCH: bool(self.branches),
        }
        if self.choice_kind in options and not options[self.choice_kind]:
            raise ValueError(f"{self.choice_kind} choice needs options")
        if self.choice_kind is ChoiceKind.BOUNDED_CARDS and self.maximum < 1:
            raise ValueError("bounded-card choices need a positive maximum")
        if self.choice_kind is not ChoiceKind.BOUNDED_CARDS and (self.minimum, self.maximum) != (
            1,
            1,
        ):
            raise ValueError("only bounded-card choices use selection bounds")
        if (
            self.order_group is not OrderGroup.ALL
            and self.choice_kind is not ChoiceKind.ORDER_CARDS
        ):
            raise ValueError("order grouping applies only to ordering choices")
        if self.owner is not None and self.choice_kind is not ChoiceKind.HIDDEN_CARD:
            raise ValueError("only a hidden-card choice names a disambiguating zone owner")
        if self.choice_kind is ChoiceKind.HIDDEN_CARD:
            assert self.cards is not None
            if self.cards.kind not in {CardSelectorKind.HAND, CardSelectorKind.SCORE}:
                raise ValueError("a hidden-card choice must address one hand or score pile")
            if self.cards.player is None or self.cards.player.is_multi:
                raise ValueError("a hidden-card choice must address exactly one zone owner")


@dataclass(frozen=True, slots=True)
class SequenceNode:
    """Execute child nodes in declared order."""

    node_id: str
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConditionNode:
    """Execute one of two branches based on an explicit predicate."""

    node_id: str
    predicate: Predicate
    when_true: str
    when_false: str | None = None


@dataclass(frozen=True, slots=True)
class RepeatNode:
    """Repeat a child after each successful predicate evaluation."""

    node_id: str
    body: str
    repeat_if: Predicate
    maximum_iterations: int = 100

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("repeat limit must be positive")


@dataclass(frozen=True, slots=True)
class BatchNode:
    """Execute mutation-only children as one externally atomic VM step."""

    node_id: str
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DrawNode:
    """Draw a card to a player's hand with upward supply fallback."""

    node_id: str
    requested_age: ValueRef
    result_variable: str
    player: PlayerRef = EXECUTOR


@dataclass(frozen=True, slots=True)
class RevealNode:
    """Emit explicit reveal provenance for cards held in a scoped variable."""

    node_id: str
    cards: CardSelector


@dataclass(frozen=True, slots=True)
class KeepNode:
    """Emit explicit keep provenance without moving the drawn card again."""

    node_id: str
    cards: CardSelector


class MovementKind(StrEnum):
    """Shared movement operations supported by one declarative node."""

    MELD = "meld"
    TUCK = "tuck"
    SCORE = "score"
    RETURN = "return"
    TRANSFER = "transfer"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class MoveNode:
    """Move selected cards through shared zone primitives.

    A transfer may target another player's board. Per rules decision 14 the card lands atop the
    matching destination stack and adopts that stack's splay, which the shared zone primitive
    already implements.
    """

    node_id: str
    movement: MovementKind
    cards: CardSelector
    destination_player: PlayerRef | None = None
    destination_zone: ZoneKind | None = None
    result_variable: str | None = None
    result_scope: VariableScope = VariableScope.LOCAL
    moved_variable: str | None = None
    order_variable: str | None = None

    def __post_init__(self) -> None:
        player_moves = {
            MovementKind.MELD,
            MovementKind.TUCK,
            MovementKind.SCORE,
            MovementKind.TRANSFER,
        }
        if (self.destination_player is not None) != (self.movement in player_moves):
            raise ValueError("player-directed movements require one destination player")
        if self.movement is MovementKind.TRANSFER:
            if self.destination_player is None or self.destination_zone not in {
                ZoneKind.HAND,
                ZoneKind.SCORE,
                ZoneKind.BOARD,
            }:
                raise ValueError("transfer requires a player hand, score, or board destination")
        elif self.destination_zone is not None:
            raise ValueError("only transfer carries an explicit destination zone")
        if self.result_variable is None and self.result_scope is not VariableScope.LOCAL:
            raise ValueError("a movement result scope requires a result variable")


@dataclass(frozen=True, slots=True)
class CollectNode:
    """Append one selected card to a deferred, duplicate-free card collection.

    Collection changes only serializable VM bookkeeping. It deliberately creates no gameplay
    event or achievement boundary, so a quantity-scaled instruction can gather every mandatory
    choice before committing one grouped movement.
    """

    node_id: str
    card_variable: str
    result_variable: str


@dataclass(frozen=True, slots=True)
class ExchangeNode:
    """Atomically exchange two selected card sets and preserve batch provenance."""

    node_id: str
    first: CardSelector
    second: CardSelector
    result_variable: str | None = None


@dataclass(frozen=True, slots=True)
class RearrangeNode:
    """Apply an ordered-card variable to one selected stack."""

    node_id: str
    player: PlayerRef
    color_variable: str
    order_variable: str
    result_variable: str | None = None


@dataclass(frozen=True, slots=True)
class SplayNode:
    """Set a stack's splay direction."""

    node_id: str
    player: PlayerRef
    color: Color | None = None
    direction: SplayDirection | None = None
    color_variable: str | None = None
    direction_variable: str | None = None
    result_variable: str | None = None

    def __post_init__(self) -> None:
        if (self.color is None) == (self.color_variable is None):
            raise ValueError("splay needs exactly one color source")
        if (self.direction is None) == (self.direction_variable is None):
            raise ValueError("splay needs exactly one direction source")


@dataclass(frozen=True, slots=True)
class RemoveAllPlayCardsNode:
    """Atomically remove both players' hands, boards, and score piles."""

    node_id: str
    result_variable: str | None = None


@dataclass(frozen=True, slots=True)
class NestedNode:
    """Execute only a selected card program's non-demand effects."""

    node_id: str
    card_variable: str


@dataclass(frozen=True, slots=True)
class AbortDogmaNode:
    """Clear the complete frame stack and suppress remaining dogma work."""

    node_id: str


@dataclass(frozen=True, slots=True)
class StopNode:
    """End this printed effect only, leaving the rest of the dogma action intact."""

    node_id: str


@dataclass(frozen=True, slots=True)
class LetNode:
    """Bind a computed quantity, card set, or colour into a scoped variable.

    Binding once, before branching, is how rules decision 17's "evaluate the quantity when the
    instruction begins" is expressed declaratively. ``color_of`` binds the colour of a card
    already held in a variable, which is what card text means by "that color".
    """

    node_id: str
    result_variable: str
    value: ValueRef | None = None
    cards: CardSelector | None = None
    color_of: str | None = None

    def __post_init__(self) -> None:
        sources = sum(source is not None for source in (self.value, self.cards, self.color_of))
        if sources != 1:
            raise ValueError("a binding needs exactly one value, card, or colour source")


@dataclass(frozen=True, slots=True)
class TimesNode:
    """Execute a child body a quantity of times, evaluated once on entry."""

    node_id: str
    count: ValueRef
    body: str
    index_variable: str | None = None
    maximum_iterations: int = 100

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("times limit must be positive")


@dataclass(frozen=True, slots=True)
class ForEachCardNode:
    """Iterate a card variable in its stored order, binding one card per pass."""

    node_id: str
    cards_variable: str
    item_variable: str
    body: str
    maximum_iterations: int = 120

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("for-each limit must be positive")


@dataclass(frozen=True, slots=True)
class AllOrNoneNode:
    """Execute one atomic body only when its complete feasibility guard holds.

    The guard must describe every prerequisite for the whole instruction; the program validator
    restricts the body to one atomic leaf or :class:`BatchNode`, so no achievement check or player
    decision can observe a partial prefix.
    """

    node_id: str
    guard: Predicate
    body: str


class ClaimRouteKind(StrEnum):
    """Which achievement route a claim node uses."""

    LINKED = "linked"


@dataclass(frozen=True, slots=True)
class ClaimAchievementNode:
    """Claim a special achievement through its linked card route."""

    node_id: str
    achievement_id: SpecialAchievementId
    player: PlayerRef = EXECUTOR
    route: ClaimRouteKind = ClaimRouteKind.LINKED
    melded_count_variable: str | None = None
    result_variable: str | None = None


class WinMode(StrEnum):
    """How a card-effect win chooses its winner."""

    EXECUTOR = "executor"
    UNIQUE_EXTREME = "unique-extreme"


class WinMetric(StrEnum):
    """Comparable quantity used by a unique-extreme win effect."""

    SCORE = "score"
    ACHIEVEMENTS = "achievements"
    VISIBLE_ICON = "visible-icon"


@dataclass(frozen=True, slots=True)
class WinNode:
    """End the game through explicit card text.

    A ``UNIQUE_EXTREME`` win is ignored entirely on a tie (rules section 11), which leaves the
    surrounding effect and the rest of the dogma action running.
    """

    node_id: str
    mode: WinMode = WinMode.EXECUTOR
    player: PlayerRef = EXECUTOR
    metric: WinMetric | None = None
    icon: Icon | None = None
    extreme: Extreme = Extreme.HIGHEST

    def __post_init__(self) -> None:
        if (self.metric is not None) != (self.mode is WinMode.UNIQUE_EXTREME):
            raise ValueError("only a unique-extreme win carries a metric")
        if (self.icon is not None) != (self.metric is WinMetric.VISIBLE_ICON):
            raise ValueError("only a visible-icon metric carries an icon")


@dataclass(frozen=True, slots=True)
class NoOpNode:
    """An explicit, serializable no-op useful for partial execution."""

    node_id: str


type EffectNode = (
    ChoiceNode
    | SequenceNode
    | ConditionNode
    | RepeatNode
    | BatchNode
    | DrawNode
    | RevealNode
    | KeepNode
    | MoveNode
    | CollectNode
    | ExchangeNode
    | RearrangeNode
    | SplayNode
    | RemoveAllPlayCardsNode
    | NestedNode
    | AbortDogmaNode
    | StopNode
    | LetNode
    | TimesNode
    | ForEachCardNode
    | AllOrNoneNode
    | ClaimAchievementNode
    | WinNode
    | NoOpNode
)


@dataclass(frozen=True, slots=True)
class ProgramEffect:
    """One printed effect entry in a declarative card program."""

    effect_id: DogmaEffectId
    demand: bool
    root_node_id: str


@dataclass(frozen=True, slots=True)
class EffectProgram:
    """A stable program keyed independently from frame progress."""

    program_id: str
    source_card_id: CardId
    effects: tuple[ProgramEffect, ...]
    nodes: tuple[EffectNode, ...]

    def __post_init__(self) -> None:
        if not self.program_id:
            raise ValueError("effect program ID cannot be empty")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("effect node IDs must be unique within a program")
        known = set(node_ids)
        if not self.effects:
            raise ValueError("effect program needs at least one effect")
        ordinals = tuple(effect.effect_id.ordinal for effect in self.effects)
        if len(set(ordinals)) != len(ordinals) or ordinals != tuple(sorted(ordinals)):
            raise ValueError("program effect ordinals must be unique and increasing")
        if any(effect.effect_id.card_id != self.source_card_id for effect in self.effects):
            raise ValueError("program effects must belong to its source card")
        if any(effect.root_node_id not in known for effect in self.effects):
            raise ValueError("program effect references an unknown root node")
        node_by_id = {node.node_id: node for node in self.nodes}
        atomic_types = (
            DrawNode,
            RevealNode,
            KeepNode,
            MoveNode,
            ExchangeNode,
            RearrangeNode,
            SplayNode,
            RemoveAllPlayCardsNode,
            NoOpNode,
        )
        for node in self.nodes:
            references: tuple[str, ...] = ()
            if isinstance(node, SequenceNode | BatchNode):
                references = node.children
            elif isinstance(node, ConditionNode):
                references = (node.when_true,) + ((node.when_false,) if node.when_false else ())
            elif isinstance(node, RepeatNode | TimesNode | ForEachCardNode | AllOrNoneNode):
                references = (node.body,)
            if any(reference not in known for reference in references):
                raise ValueError(f"node {node.node_id} references an unknown child")
            if isinstance(node, BatchNode) and any(
                not isinstance(node_by_id[child], atomic_types) for child in node.children
            ):
                raise ValueError("batch children must be atomic leaf nodes")
            if isinstance(node, AllOrNoneNode) and not isinstance(
                node_by_id[node.body], (*atomic_types, BatchNode)
            ):
                raise ValueError("all-or-none bodies must be one atomic leaf or batch")
        reachable = self.reachable_node_ids()
        unreachable = tuple(sorted(known - reachable))
        if unreachable:
            raise ValueError(f"program {self.program_id} has unreachable nodes: {unreachable}")

    def reachable_node_ids(self) -> frozenset[str]:
        """Return every node reachable from a printed effect's root node."""

        node_by_id = {node.node_id: node for node in self.nodes}
        pending = [effect.root_node_id for effect in self.effects]
        seen: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in seen or node_id not in node_by_id:
                continue
            seen.add(node_id)
            node = node_by_id[node_id]
            if isinstance(node, SequenceNode | BatchNode):
                pending.extend(node.children)
            elif isinstance(node, ConditionNode):
                pending.append(node.when_true)
                if node.when_false is not None:
                    pending.append(node.when_false)
            elif isinstance(node, RepeatNode | TimesNode | ForEachCardNode | AllOrNoneNode):
                pending.append(node.body)
        return frozenset(seen)

    def node(self, node_id: str) -> EffectNode:
        """Resolve one node by its stable local ID."""

        try:
            return next(node for node in self.nodes if node.node_id == node_id)
        except StopIteration as error:
            raise KeyError(f"unknown node {node_id!r} in program {self.program_id}") from error


class UnimplementedCardError(KeyError):
    """A card was activated but has no registered effect program.

    A missing card must fail loudly rather than behaving as a no-op, because a silent no-op is
    indistinguishable from a legal dogma action that happened to change nothing.
    """

    def __init__(self, card_id: CardId) -> None:
        self.card_id = card_id
        super().__init__(f"no effect program registered for card: {card_id}")


class EffectProgramRegistry:
    """Explicit program registry; no runtime natural-language parsing or callbacks."""

    def __init__(
        self,
        programs: tuple[EffectProgram, ...],
        *,
        predicates: Mapping[CardId, Mapping[str, NamedPredicate]] | None = None,
        values: Mapping[CardId, Mapping[str, NamedValue]] | None = None,
    ) -> None:
        if len({program.program_id for program in programs}) != len(programs):
            raise ValueError("effect program IDs must be unique")
        if len({program.source_card_id for program in programs}) != len(programs):
            raise ValueError("only one effect program may be registered per card")
        self._programs = {program.program_id: program for program in programs}
        self._by_card = {program.source_card_id: program for program in programs}
        self._predicates = {
            card_id: dict(mapping) for card_id, mapping in (predicates or {}).items()
        }
        self._values = {card_id: dict(mapping) for card_id, mapping in (values or {}).items()}
        unknown = (set(self._predicates) | set(self._values)) - set(self._by_card)
        if unknown:
            raise ValueError(
                f"named callables registered for unknown cards: {sorted(str(c) for c in unknown)}"
            )
        for predicate_mapping in self._predicates.values():
            for predicate_helper in predicate_mapping.values():
                _helper_payload(predicate_helper)
        for value_mapping in self._values.values():
            for value_helper in value_mapping.values():
                _helper_payload(value_helper)

    def program(self, program_id: str) -> EffectProgram:
        try:
            return self._programs[program_id]
        except KeyError as error:
            raise KeyError(f"unknown effect program: {program_id}") from error

    def program_for_card(self, card_id: CardId) -> EffectProgram:
        try:
            return self._by_card[card_id]
        except KeyError as error:
            raise UnimplementedCardError(card_id) from error

    def implemented_card_ids(self) -> frozenset[CardId]:
        """Return every card whose effects are registered, for wave-scoped play and fuzzing."""

        return frozenset(self._by_card)

    @property
    def programs(self) -> tuple[EffectProgram, ...]:
        """Return every registered program ordered by card ID."""

        return tuple(self._programs[key] for key in sorted(self._programs))

    def named_predicate(self, card_id: CardId, name: str) -> NamedPredicate:
        """Resolve a card module's named predicate escape hatch."""

        try:
            return self._predicates[card_id][name]
        except KeyError as error:
            raise KeyError(f"card {card_id} has no named predicate {name!r}") from error

    def named_value(self, card_id: CardId, name: str) -> NamedValue:
        """Resolve a card module's named quantity escape hatch."""

        try:
            return self._values[card_id][name]
        except KeyError as error:
            raise KeyError(f"card {card_id} has no named value {name!r}") from error

    def fingerprint(self) -> str:
        """Return a stable digest over every registered program's canonical payload.

        A behaviour change to any card changes this digest, so a WP9 log recorded against the old
        behaviour fails loudly during replay instead of silently diverging.
        """

        digest = hashlib.sha256()
        digest.update(b"innovation-ai-effect-programs-v1\0")
        for program in sorted(self.programs, key=lambda item: item.program_id):
            digest.update(program.program_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    program_payload(program),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            )
            digest.update(b"\0")
            for name, predicate_helper in sorted(
                self._predicates.get(program.source_card_id, {}).items()
            ):
                digest.update(f"predicate:{name}\0".encode())
                digest.update(_helper_payload(predicate_helper))
                digest.update(b"\0")
            for name, value_helper in sorted(self._values.get(program.source_card_id, {}).items()):
                digest.update(f"value:{name}\0".encode())
                digest.update(_helper_payload(value_helper))
                digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"


def _helper_payload(helper: NamedPredicate | NamedValue) -> bytes:
    """Return deterministic source plus captured-data bytes for one named pure helper."""

    if not inspect.isfunction(helper):
        raise TypeError("named effect helpers must be inspectable Python functions")
    function = helper
    try:
        source = textwrap.dedent(inspect.getsource(function)).strip()
    except (OSError, TypeError) as error:
        raise TypeError("named effect helpers must have inspectable Python source") from error
    closure = function.__closure__ or ()
    captured = tuple(cell.cell_contents for cell in closure)
    defaults = function.__defaults__ or ()
    metadata = {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source": source,
        "defaults": _payload(tuple(defaults)),
        "captured": _payload(captured),
    }
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def program_payload(program: EffectProgram) -> dict[str, object]:
    """Return a canonical JSON-compatible declarative program payload."""

    payload = _payload(program)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise TypeError("effect program did not serialize to an object")
    return payload


def _payload(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{
                field.name: _payload(getattr(value, field.name))
                for field in dataclass_fields(value)
            },
        }
    if isinstance(value, tuple):
        return [_payload(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"effect program contains non-serializable value: {type(value).__name__}")
