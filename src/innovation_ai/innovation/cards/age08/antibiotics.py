"""ANTIBIOTICS - optionally return up to three hand cards, then draw two 8s for
 every different value returned.

The selected subset and its distinct-value count are snapshotted before any return changes the
supply. Cards sharing an age pile are returned in the executor's chosen order.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("antibiotics")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "antibiotics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "antibiotics-effect"),),
    (
        SequenceNode("antibiotics-effect", ("choose-returns", "if-returned")),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "selected-returns",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=3,
        ),
        ConditionNode(
            "if-returned",
            Predicate.truthy("selected-returns"),
            "return-and-draw",
        ),
        SequenceNode(
            "return-and-draw",
            ("bind-distinct-values", "order-returns", "return-selected", "draw-rewards"),
        ),
        LetNode(
            "bind-distinct-values",
            "distinct-values",
            value=ValueRef(ValueRefKind.DISTINCT_VALUES, variable="selected-returns"),
        ),
        ChoiceNode(
            "order-returns",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-returns"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-selected",
            MovementKind.RETURN,
            CardSelector.from_variable("selected-returns"),
            order_variable="return-order",
        ),
        TimesNode(
            "draw-rewards",
            ValueRef.from_variable("distinct-values"),
            "draw-two-eights",
        ),
        SequenceNode("draw-two-eights", ("draw-first-eight", "draw-second-eight")),
        DrawNode("draw-first-eight", ValueRef.literal(8), "first-eight", player=EXECUTOR),
        DrawNode("draw-second-eight", ValueRef.literal(8), "second-eight", player=EXECUTOR),
    ),
)
