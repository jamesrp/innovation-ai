"""Versioned semantic actions and player-decision contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import ClassVar

from innovation_ai.innovation.observations import GameObservation
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    NormalAchievementId,
    PlayerId,
    SplayDirection,
)

ACTION_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 3
_BRANCH_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ActionKind(StrEnum):
    """Stable tags for paid actions, setup choices, and effect choices."""

    CHOOSE_STARTING_MELD = "choose-starting-meld"
    DRAW = "draw"
    MELD = "meld"
    DOGMA = "dogma"
    ACHIEVE = "achieve"
    CHOOSE_CARD = "choose-card"
    CHOOSE_CARDS = "choose-cards"
    CHOOSE_COLOR = "choose-color"
    CHOOSE_PLAYER = "choose-player"
    CHOOSE_VALUE = "choose-value"
    CHOOSE_SPLAY = "choose-splay"
    CHOOSE_BRANCH = "choose-branch"
    ORDER_CARDS = "order-cards"
    DECLINE = "decline"
    FINISH_SELECTION = "finish-selection"


class DecisionKind(StrEnum):
    """Stable semantic categories of player decision."""

    STARTING_MELD = "starting-meld"
    TURN_ACTION = "turn-action"
    EFFECT_CHOICE = "effect-choice"


@dataclass(frozen=True, slots=True)
class Action:
    """Base for actions tied to one exact decision."""

    decision_id: int
    kind: ClassVar[ActionKind]
    schema_version: ClassVar[int] = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decision_id < 1:
            raise ValueError("decision ID must be positive")


@dataclass(frozen=True, slots=True)
class ChooseStartingMeldAction(Action):
    """Secret simultaneous setup selection from the chooser's initial hand."""

    card_id: CardId
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_STARTING_MELD


@dataclass(frozen=True, slots=True)
class DrawAction(Action):
    """Take the paid Draw action."""

    kind: ClassVar[ActionKind] = ActionKind.DRAW


@dataclass(frozen=True, slots=True)
class MeldAction(Action):
    """Meld one identified card from the active player's hand."""

    card_id: CardId
    kind: ClassVar[ActionKind] = ActionKind.MELD


@dataclass(frozen=True, slots=True)
class DogmaAction(Action):
    """Activate one identified top card on the active player's board."""

    card_id: CardId
    kind: ClassVar[ActionKind] = ActionKind.DOGMA


@dataclass(frozen=True, slots=True)
class AchieveAction(Action):
    """Claim one identified, currently eligible normal achievement."""

    achievement_id: NormalAchievementId
    kind: ClassVar[ActionKind] = ActionKind.ACHIEVE


@dataclass(frozen=True, slots=True)
class ChooseCardAction(Action):
    """Choose one semantic card identity during an effect."""

    card_id: CardId
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_CARD


@dataclass(frozen=True, slots=True)
class ChooseCardsAction(Action):
    """Choose an unordered card subset, canonicalized by card ID."""

    card_ids: tuple[CardId, ...]
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_CARDS

    def __post_init__(self) -> None:
        super(ChooseCardsAction, self).__post_init__()
        if len(set(self.card_ids)) != len(self.card_ids):
            raise ValueError("a card selection cannot contain duplicates")
        object.__setattr__(self, "card_ids", tuple(sorted(self.card_ids, key=str)))


@dataclass(frozen=True, slots=True)
class ChooseColorAction(Action):
    """Choose a color during an effect."""

    color: Color
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_COLOR


@dataclass(frozen=True, slots=True)
class ChoosePlayerAction(Action):
    """Choose a player during an effect."""

    player_id: PlayerId
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_PLAYER


@dataclass(frozen=True, slots=True)
class ChooseValueAction(Action):
    """Choose an integer value during an effect."""

    value: int
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_VALUE


@dataclass(frozen=True, slots=True)
class ChooseSplayAction(Action):
    """Choose a splay direction during an effect."""

    direction: SplayDirection
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_SPLAY


@dataclass(frozen=True, slots=True)
class ChooseBranchAction(Action):
    """Choose a stable implementation-defined branch identifier."""

    branch_id: str
    kind: ClassVar[ActionKind] = ActionKind.CHOOSE_BRANCH

    def __post_init__(self) -> None:
        super(ChooseBranchAction, self).__post_init__()
        if _BRANCH_ID.fullmatch(self.branch_id) is None:
            raise ValueError(f"invalid semantic branch ID: {self.branch_id!r}")


@dataclass(frozen=True, slots=True)
class OrderCardsAction(Action):
    """Choose an authoritative order of identified cards."""

    card_ids: tuple[CardId, ...]
    kind: ClassVar[ActionKind] = ActionKind.ORDER_CARDS

    def __post_init__(self) -> None:
        super(OrderCardsAction, self).__post_init__()
        if len(set(self.card_ids)) != len(self.card_ids):
            raise ValueError("a card ordering cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class DeclineAction(Action):
    """Decline an explicitly optional instruction."""

    kind: ClassVar[ActionKind] = ActionKind.DECLINE


@dataclass(frozen=True, slots=True)
class FinishSelectionAction(Action):
    """Stop an incremental bounded selection."""

    kind: ClassVar[ActionKind] = ActionKind.FINISH_SELECTION


type SemanticAction = (
    ChooseStartingMeldAction
    | DrawAction
    | MeldAction
    | DogmaAction
    | AchieveAction
    | ChooseCardAction
    | ChooseCardsAction
    | ChooseColorAction
    | ChoosePlayerAction
    | ChooseValueAction
    | ChooseSplayAction
    | ChooseBranchAction
    | OrderCardsAction
    | DeclineAction
    | FinishSelectionAction
)


@dataclass(frozen=True, slots=True)
class DecisionSource:
    """Optional card/effect provenance for a decision."""

    card_id: CardId
    effect_id: DogmaEffectId | None = None

    def __post_init__(self) -> None:
        if self.effect_id is not None and self.effect_id.card_id != self.card_id:
            raise ValueError("decision source effect must belong to its source card")


class IncrementalSelectionKind(StrEnum):
    """Semantic purpose of repeated choose-next decisions inside one effect choice."""

    NONE = "none"
    BOUNDED_SUBSET = "bounded-subset"
    CARD_ORDER = "card-order"


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Dogma and selection context a policy needs to read a decision's semantics.

    Without this, a forced demand compliance and a voluntary choice are indistinguishable in a
    log or to a policy encoder, because both arrive as "choose one of these cards".
    """

    demand: bool = False
    shared: bool = False
    nested: bool = False
    featured_icon: Icon | None = None
    activator_icons: int | None = None
    opponent_icons: int | None = None
    minimum_count: int = 1
    maximum_count: int = 1
    selected_so_far: tuple[CardId, ...] = ()
    incremental_selection: IncrementalSelectionKind = IncrementalSelectionKind.NONE

    def __post_init__(self) -> None:
        if self.minimum_count < 0 or self.maximum_count < self.minimum_count:
            raise ValueError("invalid decision selection bounds")
        if (self.activator_icons is None) != (self.opponent_icons is None):
            raise ValueError("frozen icon counts must be supplied together")
        if self.activator_icons is not None and self.activator_icons < 0:
            raise ValueError("frozen icon counts cannot be negative")
        if self.opponent_icons is not None and self.opponent_icons < 0:
            raise ValueError("frozen icon counts cannot be negative")
        if len(set(self.selected_so_far)) != len(self.selected_so_far):
            raise ValueError("an incremental selection cannot repeat a card")


@dataclass(frozen=True, slots=True)
class Decision:
    """One player-safe, deterministic set of legal semantic actions."""

    decision_id: int
    kind: DecisionKind
    chooser: PlayerId
    executor: PlayerId
    observation: GameObservation
    legal_actions: tuple[SemanticAction, ...]
    source: DecisionSource | None = None
    dogma_activator: PlayerId | None = None
    dogma_action_id: int | None = None
    context: DecisionContext | None = None
    schema_version: int = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decision_id < 1:
            raise ValueError("decision ID must be positive")
        if self.dogma_action_id is not None and self.dogma_action_id < 1:
            raise ValueError("dogma action ID must be positive")
        if (self.dogma_activator is None) != (self.dogma_action_id is None):
            raise ValueError("dogma activator and action ID must be supplied together")
        if not self.legal_actions:
            raise ValueError("a decision must contain at least one legal action")
        if any(action.decision_id != self.decision_id for action in self.legal_actions):
            raise ValueError("every legal action must reference its decision")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal actions cannot contain duplicates")


def _payload_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _payload_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_payload_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"contract contains non-serializable value: {type(value).__name__}")


def action_payload(action: SemanticAction) -> dict[str, object]:
    """Return the canonical JSON-compatible action payload."""

    payload = {field.name: _payload_value(getattr(action, field.name)) for field in fields(action)}
    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "kind": action.kind.value,
        **payload,
    }


def decision_payload(decision: Decision) -> dict[str, object]:
    """Return the canonical JSON-compatible decision payload."""

    return {
        "schema_version": decision.schema_version,
        "decision_id": decision.decision_id,
        "kind": decision.kind.value,
        "chooser": decision.chooser.value,
        "executor": decision.executor.value,
        "source": _payload_value(decision.source),
        "dogma_activator": _payload_value(decision.dogma_activator),
        "dogma_action_id": decision.dogma_action_id,
        "context": _payload_value(decision.context),
        "observation": _payload_value(decision.observation),
        "legal_actions": [action_payload(action) for action in decision.legal_actions],
    }
