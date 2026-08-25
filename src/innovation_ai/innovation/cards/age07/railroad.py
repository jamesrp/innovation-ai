"""RAILROAD - "Return all cards from your hand, then draw three 6." Then: "You may
splay up any one color of your cards currently splayed right."

The full hand is bound before movement and the returning player orders cards sharing an age pile.
For the second effect, a pure selector predicate identifies the public top card of each currently
right-splayed stack; that stable card ID is used as the declarative proxy for its colour.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("railroad")


def _candidate_is_splayed_right(state: Any, context: Any, registry: Any) -> bool:
    """Whether the selector's candidate is atop a stack currently splayed right."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    if not isinstance(raw_candidate, str):
        return False
    color = registry.card(CardId(raw_candidate)).color
    return state.player(context.executor).board.stack(color).splay is SplayDirection.RIGHT


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-splayed-right": _candidate_is_splayed_right,
}

_RIGHT_SPLAYED_TOPS: Final[CardSelector] = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    predicate="candidate-is-splayed-right",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "railroad-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "railroad-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "railroad-splay"),
    ),
    (
        SequenceNode(
            "railroad-return",
            ("bind-hand", "order-hand", "return-hand", "draw-three-sixes"),
        ),
        LetNode("bind-hand", "returned-hand", cards=CardSelector.hand(EXECUTOR)),
        ChoiceNode(
            "order-hand",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("returned-hand"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-hand",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-hand"),
            order_variable="return-order",
        ),
        TimesNode("draw-three-sixes", ValueRef.literal(3), "draw-six"),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        SequenceNode("railroad-splay", ("choose-right-stack", "if-right-stack")),
        ChoiceNode(
            "choose-right-stack",
            ChoiceKind.CARD,
            "right-stack-top",
            chooser=EXECUTOR,
            cards=_RIGHT_SPLAYED_TOPS,
            optional=True,
        ),
        ConditionNode(
            "if-right-stack",
            Predicate.truthy("right-stack-top"),
            "splay-up-sequence",
        ),
        SequenceNode("splay-up-sequence", ("bind-right-color", "splay-right-color-up")),
        LetNode("bind-right-color", "right-color", color_of="right-stack-top"),
        SplayNode(
            "splay-right-color-up",
            EXECUTOR,
            color_variable="right-color",
            direction=SplayDirection.UP,
            result_variable="splayed",
        ),
    ),
)
