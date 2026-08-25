"""VACCINATION - return all lowest score cards, then resolve its two causal rewards.

The direct demand stores whether its grouped return moved anything in this exact card execution's
causal scope. Nested execution skips the demand and therefore cannot inherit unrelated outer
history.
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
    Extreme,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    VariableScope,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("vaccination")


EFFECTS: Final[EffectProgram] = EffectProgram(
    "vaccination-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "vaccination-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "vaccination-follow-up"),
    ),
    (
        SequenceNode(
            "vaccination-demand",
            (
                "snapshot-lowest",
                "order-lowest",
                "return-lowest",
                "if-returned-lowest",
            ),
        ),
        LetNode(
            "snapshot-lowest",
            "lowest-cards",
            cards=CardSelector.score(EXECUTOR, extreme=Extreme.LOWEST),
        ),
        ChoiceNode(
            "order-lowest",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("lowest-cards"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-lowest",
            MovementKind.RETURN,
            CardSelector.from_variable("lowest-cards"),
            result_variable="vaccination-demand-returned",
            result_scope=VariableScope.CARD_EXECUTION,
            moved_variable="returned-lowest",
            order_variable="return-order",
        ),
        ConditionNode(
            "if-returned-lowest",
            Predicate.truthy("returned-lowest"),
            "draw-and-meld-six",
        ),
        BatchNode("draw-and-meld-six", ("draw-six", "meld-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        MoveNode(
            "meld-six",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-six"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "vaccination-follow-up",
            Predicate.truthy(
                "vaccination-demand-returned",
                scope=VariableScope.CARD_EXECUTION,
            ),
            "draw-and-meld-seven",
        ),
        BatchNode("draw-and-meld-seven", ("draw-seven", "meld-seven")),
        DrawNode("draw-seven", ValueRef.literal(7), "drawn-seven", player=EXECUTOR),
        MoveNode(
            "meld-seven",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-seven"),
            destination_player=EXECUTOR,
        ),
    ),
)
