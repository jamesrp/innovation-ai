"""LIGHTING - "You may tuck up to three cards from your hand. If you do, draw and score
a 7 for every different value of card you tucked."

Subset selection is intentionally separate from same-colour tuck ordering. The distinct-value
reward is bound from the selected subset before movement and then consumed as one fixed quantity.
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

CARD_ID: Final[CardId] = CardId("lighting")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "lighting-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "lighting-effect"),),
    (
        SequenceNode("lighting-effect", ("choose-tucks", "if-tucked")),
        ChoiceNode(
            "choose-tucks",
            ChoiceKind.BOUNDED_CARDS,
            "selected-tucks",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=3,
        ),
        ConditionNode(
            "if-tucked",
            Predicate.truthy("selected-tucks"),
            "tuck-and-reward",
        ),
        SequenceNode(
            "tuck-and-reward",
            ("bind-distinct", "order-tucks", "tuck-selected", "score-sevens"),
        ),
        LetNode(
            "bind-distinct",
            "distinct-values",
            value=ValueRef(ValueRefKind.DISTINCT_VALUES, variable="selected-tucks"),
        ),
        ChoiceNode(
            "order-tucks",
            ChoiceKind.ORDER_CARDS,
            "tuck-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-tucks"),
            order_group=OrderGroup.COLOR,
        ),
        MoveNode(
            "tuck-selected",
            MovementKind.TUCK,
            CardSelector.from_variable("selected-tucks"),
            destination_player=EXECUTOR,
            order_variable="tuck-order",
            moved_variable="tucked",
        ),
        TimesNode(
            "score-sevens",
            ValueRef.from_variable("distinct-values"),
            "draw-and-score-seven",
        ),
        SequenceNode("draw-and-score-seven", ("draw-seven", "score-seven")),
        DrawNode("draw-seven", ValueRef.literal(7), "drawn-seven", player=EXECUTOR),
        MoveNode(
            "score-seven",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-seven"),
            destination_player=EXECUTOR,
        ),
    ),
)
