"""Declarative effect programs interpreted by the resumable VM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    PlayerId,
    SplayDirection,
)
from innovation_ai.innovation.zones import ZoneKind


class PlayerRefKind(StrEnum):
    """Ways an effect can identify a player without a callback."""

    ACTOR = "actor"
    CHOOSER = "chooser"
    EXECUTOR = "executor"
    ACTIVATOR = "activator"
    OPPONENT_OF_EXECUTOR = "opponent-of-executor"
    LITERAL = "literal"
    VARIABLE = "variable"


@dataclass(frozen=True, slots=True)
class PlayerRef:
    """Serializable player reference used by choices and movements."""

    kind: PlayerRefKind
    player_id: PlayerId | None = None
    variable: str | None = None

    def __post_init__(self) -> None:
        if (self.player_id is not None) != (self.kind is PlayerRefKind.LITERAL):
            raise ValueError("only a literal player reference carries a player ID")
        if (self.variable is not None) != (self.kind is PlayerRefKind.VARIABLE):
            raise ValueError("only a variable player reference carries a variable name")

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


class CardSelectorKind(StrEnum):
    """Supported deterministic card-set queries."""

    HAND = "hand"
    SCORE = "score"
    BOARD_STACK = "board-stack"
    TOP_CARDS = "top-cards"
    VARIABLE = "variable"
    CONSTANT = "constant"


@dataclass(frozen=True, slots=True)
class CardSelector:
    """A declarative card query evaluated against live authoritative state."""

    kind: CardSelectorKind
    player: PlayerRef | None = None
    color: Color | None = None
    color_variable: str | None = None
    variable: str | None = None
    cards: tuple[CardId, ...] = ()
    icon: Icon | None = None
    highest_only: bool = False
    exclude_source_card: bool = False

    def __post_init__(self) -> None:
        player_kinds = {
            CardSelectorKind.HAND,
            CardSelectorKind.SCORE,
            CardSelectorKind.BOARD_STACK,
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

    @classmethod
    def hand(
        cls, player: PlayerRef = EXECUTOR, *, icon: Icon | None = None, highest_only: bool = False
    ) -> CardSelector:
        return cls(CardSelectorKind.HAND, player, icon=icon, highest_only=highest_only)

    @classmethod
    def top_cards(
        cls, player: PlayerRef = EXECUTOR, *, exclude_source_card: bool = False
    ) -> CardSelector:
        return cls(
            CardSelectorKind.TOP_CARDS,
            player,
            exclude_source_card=exclude_source_card,
        )

    @classmethod
    def stack(
        cls,
        player: PlayerRef = EXECUTOR,
        *,
        color: Color | None = None,
        color_variable: str | None = None,
    ) -> CardSelector:
        return cls(
            CardSelectorKind.BOARD_STACK,
            player,
            color=color,
            color_variable=color_variable,
        )

    @classmethod
    def from_variable(cls, variable: str) -> CardSelector:
        return cls(CardSelectorKind.VARIABLE, variable=variable)


class ValueRefKind(StrEnum):
    """Supported integer expression forms."""

    LITERAL = "literal"
    VARIABLE = "variable"
    COUNT_CARDS = "count-cards"


@dataclass(frozen=True, slots=True)
class ValueRef:
    """A small serializable integer expression."""

    kind: ValueRefKind
    value: int | None = None
    variable: str | None = None

    def __post_init__(self) -> None:
        if (self.value is not None) != (self.kind is ValueRefKind.LITERAL):
            raise ValueError("only a literal value reference carries a value")
        if (self.variable is not None) != (
            self.kind in {ValueRefKind.VARIABLE, ValueRefKind.COUNT_CARDS}
        ):
            raise ValueError("variable/count references need a variable name")

    @classmethod
    def literal(cls, value: int) -> ValueRef:
        return cls(ValueRefKind.LITERAL, value=value)

    @classmethod
    def count(cls, variable: str) -> ValueRef:
        return cls(ValueRefKind.COUNT_CARDS, variable=variable)


class PredicateKind(StrEnum):
    """Supported conditions for branches and repeats."""

    VARIABLE_TRUTHY = "variable-truthy"
    VARIABLE_EQUALS = "variable-equals"
    CARD_HAS_ICON = "card-has-icon"
    CARD_COLOR_IS = "card-color-is"


@dataclass(frozen=True, slots=True)
class Predicate:
    """A deterministic condition over scoped variables and catalog facts."""

    kind: PredicateKind
    variable: str
    icon: Icon | None = None
    color: Color | None = None
    expected: str | int | bool | None = None

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("predicate variable cannot be empty")
        if (self.icon is not None) != (self.kind is PredicateKind.CARD_HAS_ICON):
            raise ValueError("only an icon predicate carries an icon")
        if (self.color is not None) != (self.kind is PredicateKind.CARD_COLOR_IS):
            raise ValueError("only a color predicate carries a color")
        if (self.expected is not None) != (self.kind is PredicateKind.VARIABLE_EQUALS):
            raise ValueError("only an equality predicate carries an expected value")

    @classmethod
    def truthy(cls, variable: str) -> Predicate:
        return cls(PredicateKind.VARIABLE_TRUTHY, variable)

    @classmethod
    def equals(cls, variable: str, expected: str | int | bool) -> Predicate:
        return cls(PredicateKind.VARIABLE_EQUALS, variable, expected=expected)

    @classmethod
    def card_has_icon(cls, variable: str, icon: Icon) -> Predicate:
        return cls(PredicateKind.CARD_HAS_ICON, variable, icon=icon)

    @classmethod
    def card_color_is(cls, variable: str, color: Color) -> Predicate:
        return cls(PredicateKind.CARD_COLOR_IS, variable, color=color)


class ChoiceKind(StrEnum):
    """Effect choice shapes mapped to WP3 semantic actions."""

    CARD = "card"
    BOUNDED_CARDS = "bounded-cards"
    COLOR = "color"
    PLAYER = "player"
    VALUE = "value"
    SPLAY = "splay"
    BRANCH = "branch"
    ORDER_CARDS = "order-cards"


@dataclass(frozen=True, slots=True)
class ChoiceNode:
    """Pause for a deterministic first-class effect choice."""

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
    only_effective_return_order: bool = False
    minimum_stack_size: int = 0

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("invalid choice bounds")
        card_kinds = {ChoiceKind.CARD, ChoiceKind.BOUNDED_CARDS, ChoiceKind.ORDER_CARDS}
        if (self.cards is not None) != (self.choice_kind in card_kinds):
            raise ValueError("card-shaped choices require exactly one card selector")
        options = {
            ChoiceKind.COLOR: bool(self.colors),
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
        if self.only_effective_return_order and self.choice_kind is not ChoiceKind.ORDER_CARDS:
            raise ValueError("return-order filtering applies only to ordering choices")


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
    """Move selected cards through shared zone primitives."""

    node_id: str
    movement: MovementKind
    cards: CardSelector
    destination_player: PlayerRef | None = None
    destination_zone: ZoneKind | None = None
    result_variable: str | None = None

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
            }:
                raise ValueError("transfer requires a player hand or score destination")
        elif self.destination_zone is not None:
            raise ValueError("only transfer carries an explicit destination zone")


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
    | ExchangeNode
    | RearrangeNode
    | SplayNode
    | RemoveAllPlayCardsNode
    | NestedNode
    | AbortDogmaNode
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
            elif isinstance(node, RepeatNode):
                references = (node.body,)
            if any(reference not in known for reference in references):
                raise ValueError(f"node {node.node_id} references an unknown child")
            if isinstance(node, BatchNode) and any(
                not isinstance(node_by_id[child], atomic_types) for child in node.children
            ):
                raise ValueError("batch children must be atomic leaf nodes")

    def node(self, node_id: str) -> EffectNode:
        """Resolve one node by its stable local ID."""

        try:
            return next(node for node in self.nodes if node.node_id == node_id)
        except StopIteration as error:
            raise KeyError(f"unknown node {node_id!r} in program {self.program_id}") from error


class EffectProgramRegistry:
    """Explicit program registry; no runtime natural-language parsing or callbacks."""

    def __init__(self, programs: tuple[EffectProgram, ...]) -> None:
        if len({program.program_id for program in programs}) != len(programs):
            raise ValueError("effect program IDs must be unique")
        if len({program.source_card_id for program in programs}) != len(programs):
            raise ValueError("only one effect program may be registered per card")
        self._programs = {program.program_id: program for program in programs}
        self._by_card = {program.source_card_id: program for program in programs}

    def program(self, program_id: str) -> EffectProgram:
        try:
            return self._programs[program_id]
        except KeyError as error:
            raise KeyError(f"unknown effect program: {program_id}") from error

    def program_for_card(self, card_id: CardId) -> EffectProgram:
        try:
            return self._by_card[card_id]
        except KeyError as error:
            raise KeyError(f"no effect program registered for card: {card_id}") from error
