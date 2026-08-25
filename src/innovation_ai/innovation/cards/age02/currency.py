"""CURRENCY - "You may return any number of cards from your hand. If you do, draw and
score a 2 for every different value of card you returned."

Subset construction is canonical, while same-age return order is a separate owner choice.  The
reward count uses the cards that actually moved and is snapshotted once when the repeated reward
instruction begins; each draw-and-score is atomic.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
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
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("currency")

_DISTINCT_RETURNED_VALUES: Final[ValueRef] = ValueRef(
    ValueRefKind.DISTINCT_VALUES,
    variable="cards-returned",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "currency-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "currency-effect"),),
    (
        SequenceNode("currency-effect", ("choose-returns", "if-selected")),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "selected-returns",
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=105,
        ),
        ConditionNode(
            "if-selected",
            Predicate.truthy("selected-returns"),
            "return-and-reward",
        ),
        SequenceNode(
            "return-and-reward",
            ("order-returns", "return-cards", "reward-by-distinct-value"),
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
            "return-cards",
            MovementKind.RETURN,
            CardSelector.from_variable("selected-returns"),
            moved_variable="cards-returned",
            order_variable="return-order",
        ),
        TimesNode(
            "reward-by-distinct-value",
            _DISTINCT_RETURNED_VALUES,
            "draw-score-two",
            maximum_iterations=10,
        ),
        BatchNode("draw-score-two", ("draw-two", "score-two")),
        DrawNode("draw-two", ValueRef.literal(2), "reward-card", player=EXECUTOR),
        MoveNode(
            "score-two",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
