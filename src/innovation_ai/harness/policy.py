"""Framework-free public contracts for learned value policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from innovation_ai.innovation.actions import (
    Decision,
    DecisionContext,
    DecisionKind,
    DecisionSource,
    IncrementalSelectionKind,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import GameObservation, observe
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import CardId, Icon, PlayerId

VALUE_POSITION_SCHEMA_VERSION = 1
PUBLIC_BOUNDARY_SCHEMA_VERSION = 1


class BatchValueEvaluator(Protocol):
    """Tensor- and framework-free batch value boundary."""

    def evaluate(self, positions: Sequence[ValuePosition], /) -> tuple[float, ...]:
        """Return one viewer-relative value for each supplied position."""


@dataclass(frozen=True, slots=True)
class CandidateRoute:
    """Route one flattened candidate value back to a semantic action group."""

    game_id: str
    decision_id: int
    action: SemanticAction
    sample_index: int
    evaluator_key: str

    def __post_init__(self) -> None:
        if not self.game_id or not self.evaluator_key:
            raise ValueError("candidate route IDs cannot be empty")
        if self.decision_id < 1 or self.sample_index < 0:
            raise ValueError("candidate route indices are invalid")
        if self.action.decision_id != self.decision_id:
            raise ValueError("candidate route action and decision IDs differ")


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """Auditable learned-policy selection from one candidate group."""

    policy_id: str
    game_id: str
    decision_id: int
    action: SemanticAction
    mean_value: float
    temperature: float

    def __post_init__(self) -> None:
        if not self.policy_id or not self.game_id:
            raise ValueError("policy selection IDs cannot be empty")
        if self.action.decision_id != self.decision_id:
            raise ValueError("policy selection action and decision IDs differ")
        if not 0.0 <= self.mean_value <= 1.0:
            raise ValueError("policy selection value must be in [0, 1]")
        if self.temperature < 0.0:
            raise ValueError("policy selection temperature cannot be negative")


class PlayerRelation(StrEnum):
    """A canonical-seat-free relationship to the encoded viewer."""

    SELF = "self"
    OPPONENT = "opponent"
    NONE = "none"


class ValuePositionKind(StrEnum):
    """Whether a position is a live boundary or a hypothetical post-action boundary."""

    CURRENT = "current"
    AFTERSTATE = "afterstate"


@dataclass(frozen=True, slots=True)
class PublicTurnProgress:
    """Viewpoint-relative public movement counters for Monument eligibility."""

    self_tucked: int
    self_scored: int
    opponent_tucked: int
    opponent_scored: int

    def __post_init__(self) -> None:
        if (
            min(
                self.self_tucked,
                self.self_scored,
                self.opponent_tucked,
                self.opponent_scored,
            )
            < 0
        ):
            raise ValueError("public turn progress cannot be negative")


@dataclass(frozen=True, slots=True)
class PublicDecisionContext:
    """A viewer-sanitized decision context with explicit unknown selection data."""

    demand: bool
    shared: bool
    nested: bool
    featured_icon: Icon | None
    activator_icons: int | None
    opponent_icons: int | None
    minimum_count: int
    maximum_count: int
    visible_selected_cards: tuple[CardId, ...]
    unknown_selected_count: int
    incremental_selection: IncrementalSelectionKind

    def __post_init__(self) -> None:
        if self.minimum_count < 0 or self.maximum_count < self.minimum_count:
            raise ValueError("invalid public decision selection bounds")
        if (self.activator_icons is None) != (self.opponent_icons is None):
            raise ValueError("public frozen icon counts must be supplied together")
        if self.activator_icons is not None and self.activator_icons < 0:
            raise ValueError("public frozen icon counts cannot be negative")
        if self.opponent_icons is not None and self.opponent_icons < 0:
            raise ValueError("public frozen icon counts cannot be negative")
        if self.unknown_selected_count < 0:
            raise ValueError("unknown selected-card count cannot be negative")
        if len(set(self.visible_selected_cards)) != len(self.visible_selected_cards):
            raise ValueError("visible selected cards cannot repeat")


@dataclass(frozen=True, slots=True)
class PublicBoundary:
    """Public protocol metadata at a value-evaluation boundary."""

    decision_kind: DecisionKind | None
    chooser_relation: PlayerRelation
    executor_relation: PlayerRelation
    dogma_activator_relation: PlayerRelation
    source: DecisionSource | None
    context: PublicDecisionContext | None
    turn_progress: PublicTurnProgress
    schema_version: int = PUBLIC_BOUNDARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PUBLIC_BOUNDARY_SCHEMA_VERSION:
            raise ValueError(f"unsupported public-boundary schema {self.schema_version}")
        if self.decision_kind is None:
            if self.chooser_relation is not PlayerRelation.NONE:
                raise ValueError("a boundary without a decision cannot have a chooser")
            if self.executor_relation is not PlayerRelation.NONE:
                raise ValueError("a boundary without a decision cannot have an executor")


@dataclass(frozen=True, slots=True)
class ValuePosition:
    """A detached, tensor-free position supplied to a value evaluator."""

    viewer: PlayerId
    observation: GameObservation
    boundary: PublicBoundary
    position_kind: ValuePositionKind = ValuePositionKind.CURRENT
    schema_version: int = VALUE_POSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.observation.viewer is not self.viewer:
            raise ValueError("value-position viewer differs from its observation viewer")
        if self.schema_version != VALUE_POSITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported value-position schema {self.schema_version}")


def player_relation(player_id: PlayerId | None, viewer: PlayerId) -> PlayerRelation:
    """Return a viewpoint-relative player relationship."""

    if player_id is None:
        return PlayerRelation.NONE
    return PlayerRelation.SELF if player_id is viewer else PlayerRelation.OPPONENT


def _visible_card_ids(observation: GameObservation) -> frozenset[CardId]:
    visible = set(observation.revealed_cards)
    for player in observation.players:
        visible.update(player.hand.known_cards)
        visible.update(player.score_pile.known_cards)
        for stack in player.board:
            if stack.top_card_id is not None:
                visible.add(stack.top_card_id)
            visible.update(
                covered.card_id for covered in stack.covered_cards if covered.card_id is not None
            )
    return frozenset(visible)


def sanitize_decision_context(
    context: DecisionContext | None,
    observation: GameObservation,
) -> PublicDecisionContext | None:
    """Remove selected-card identities that are not visible in ``observation``."""

    if context is None:
        return None
    visible_ids = _visible_card_ids(observation)
    visible_selected = tuple(
        card_id for card_id in context.selected_so_far if card_id in visible_ids
    )
    return PublicDecisionContext(
        demand=context.demand,
        shared=context.shared,
        nested=context.nested,
        featured_icon=context.featured_icon,
        activator_icons=context.activator_icons,
        opponent_icons=context.opponent_icons,
        minimum_count=context.minimum_count,
        maximum_count=context.maximum_count,
        visible_selected_cards=visible_selected,
        unknown_selected_count=len(context.selected_so_far) - len(visible_selected),
        incremental_selection=context.incremental_selection,
    )


def _turn_progress(state: GameState, viewer: PlayerId) -> PublicTurnProgress:
    other = next(player for player in PlayerId if player is not viewer)
    own = state.turn_counters.for_player(viewer)
    opposing = state.turn_counters.for_player(other)
    return PublicTurnProgress(own.tucked, own.scored, opposing.tucked, opposing.scored)


def public_boundary(
    state: GameState,
    viewer: PlayerId,
    decision: Decision | None,
    observation: GameObservation,
) -> PublicBoundary:
    """Build audited public protocol metadata for ``viewer``."""

    if observation.viewer is not viewer:
        raise ValueError("boundary observation must belong to its viewer")
    if decision is None:
        return PublicBoundary(
            decision_kind=None,
            chooser_relation=PlayerRelation.NONE,
            executor_relation=PlayerRelation.NONE,
            dogma_activator_relation=PlayerRelation.NONE,
            source=None,
            context=None,
            turn_progress=_turn_progress(state, viewer),
        )
    return PublicBoundary(
        decision_kind=decision.kind,
        chooser_relation=player_relation(decision.chooser, viewer),
        executor_relation=player_relation(decision.executor, viewer),
        dogma_activator_relation=player_relation(decision.dogma_activator, viewer),
        source=decision.source,
        context=sanitize_decision_context(decision.context, observation),
        turn_progress=_turn_progress(state, viewer),
    )


def build_value_position(
    state: GameState,
    viewer: PlayerId,
    decision: Decision | None,
    *,
    position_kind: ValuePositionKind,
    registry: CardRegistry | None = None,
) -> ValuePosition:
    """Build a fresh original-viewer position without reusing another chooser's observation."""

    registry = registry or load_card_registry()
    observation = observe(state, viewer, registry)
    return ValuePosition(
        viewer,
        observation,
        public_boundary(state, viewer, decision, observation),
        position_kind,
    )


def build_current_value_position(
    state: GameState,
    decision: Decision,
    registry: CardRegistry | None = None,
) -> ValuePosition:
    """Build the value position for a currently pending decision's chooser."""

    registry = registry or load_card_registry()
    fresh = build_value_position(
        state,
        decision.chooser,
        decision,
        position_kind=ValuePositionKind.CURRENT,
        registry=registry,
    )
    if fresh.observation != decision.observation:
        raise ValueError("decision observation does not match the supplied authoritative boundary")
    return fresh


def build_afterstate_value_position(
    state: GameState,
    original_viewer: PlayerId,
    next_decision: Decision | None,
    registry: CardRegistry | None = None,
) -> ValuePosition:
    """Build a hypothetical position from the action chooser's fresh post-action view."""

    return build_value_position(
        state,
        original_viewer,
        next_decision,
        position_kind=ValuePositionKind.AFTERSTATE,
        registry=registry,
    )
