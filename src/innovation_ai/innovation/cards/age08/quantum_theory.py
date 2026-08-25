"""QUANTUM THEORY - optionally return up to two hand cards; exactly two returns
cause one 10 to be drawn to hand followed by a second 10 drawn and scored.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
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
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("quantum-theory")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "quantum-theory-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "quantum-theory-effect"),),
    (
        SequenceNode(
            "quantum-theory-effect",
            ("choose-returns", "bind-return-count", "order-returns", "return-selected", "if-two"),
        ),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "selected-returns",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=2,
        ),
        LetNode(
            "bind-return-count",
            "return-count",
            value=ValueRef.count("selected-returns"),
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
        ConditionNode(
            "if-two",
            Predicate.count(
                ValueRef.from_variable("return-count"),
                Cmp.EQ,
                ValueRef.literal(2),
            ),
            "draw-then-draw-and-score",
        ),
        SequenceNode(
            "draw-then-draw-and-score",
            ("draw-first-ten", "draw-second-ten", "score-second-ten"),
        ),
        DrawNode("draw-first-ten", ValueRef.literal(10), "first-ten", player=EXECUTOR),
        DrawNode("draw-second-ten", ValueRef.literal(10), "second-ten", player=EXECUTOR),
        MoveNode(
            "score-second-ten",
            MovementKind.SCORE,
            CardSelector.from_variable("second-ten"),
            destination_player=EXECUTOR,
        ),
    ),
)
