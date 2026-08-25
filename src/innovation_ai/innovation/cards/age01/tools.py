"""TOOLS - effect 1: "You may return three cards from your hand. If you do, draw and meld
a 3." effect 2: "You may return a 3 from your hand. If you do, draw three 1."

Effect one first checks that an exact set of three is feasible, then separates its optional branch,
canonical subset, and meaningful return order.  Its reward is atomic draw-and-meld.  Effect two
uses an optional value-filtered card choice followed by exactly three independent draws.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("tools")

_HAS_THREE_CARDS: Final[Predicate] = Predicate.count(
    ValueRef.count_selector(CardSelector.hand(EXECUTOR)),
    Cmp.GE,
    ValueRef.literal(3),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "tools-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "tools-return-three"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "tools-return-three-value"),
    ),
    (
        ConditionNode("tools-return-three", _HAS_THREE_CARDS, "offer-return-three"),
        SequenceNode("offer-return-three", ("offer-three", "if-accepted")),
        ChoiceNode(
            "offer-three",
            ChoiceKind.BRANCH,
            "return-three-accepted",
            branches=("return-three",),
            optional=True,
        ),
        ConditionNode(
            "if-accepted",
            Predicate.truthy("return-three-accepted"),
            "return-three-sequence",
        ),
        SequenceNode(
            "return-three-sequence",
            ("choose-three", "order-three", "return-three", "draw-meld-three"),
        ),
        ChoiceNode(
            "choose-three",
            ChoiceKind.BOUNDED_CARDS,
            "three-cards",
            cards=CardSelector.hand(EXECUTOR),
            minimum=3,
            maximum=3,
        ),
        ChoiceNode(
            "order-three",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("three-cards"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-three",
            MovementKind.RETURN,
            CardSelector.from_variable("three-cards"),
            order_variable="return-order",
        ),
        BatchNode("draw-meld-three", ("draw-three", "meld-three")),
        DrawNode("draw-three", ValueRef.literal(3), "drawn-three", player=EXECUTOR),
        MoveNode(
            "meld-three",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-three"),
            destination_player=EXECUTOR,
        ),
        SequenceNode(
            "tools-return-three-value",
            ("choose-value-three", "if-value-three"),
        ),
        ChoiceNode(
            "choose-value-three",
            ChoiceKind.CARD,
            "returned-three",
            cards=CardSelector.hand(EXECUTOR, value=3),
            optional=True,
        ),
        ConditionNode(
            "if-value-three",
            Predicate.truthy("returned-three"),
            "return-and-draw-ones",
        ),
        SequenceNode(
            "return-and-draw-ones",
            ("return-value-three", "draw-three-ones"),
        ),
        MoveNode(
            "return-value-three",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-three"),
            result_variable="did-return-three",
        ),
        TimesNode("draw-three-ones", ValueRef.literal(3), "draw-one"),
        DrawNode("draw-one", ValueRef.literal(1), "drawn-one", player=EXECUTOR),
    ),
)
