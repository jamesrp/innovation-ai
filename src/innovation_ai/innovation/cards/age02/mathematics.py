"""MATHEMATICS - "You may return a card from your hand. If you do, draw and meld a card
of value one higher than the card you returned."

The returned value is bound before movement.  Declining or lacking a hand card is a no-op, while
the reward's draw-and-meld is one atomic instruction.
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
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("mathematics")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mathematics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "mathematics-effect"),),
    (
        SequenceNode("mathematics-effect", ("choose-return", "if-returned")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        ConditionNode(
            "if-returned",
            Predicate.truthy("returned-card"),
            "return-and-meld",
        ),
        SequenceNode(
            "return-and-meld",
            ("bind-reward-value", "return-card", "draw-meld-reward"),
        ),
        LetNode(
            "bind-reward-value",
            "reward-value",
            value=ValueRef.card_value("returned-card", offset=1),
        ),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
        ),
        BatchNode("draw-meld-reward", ("draw-reward", "meld-reward")),
        DrawNode(
            "draw-reward",
            ValueRef.from_variable("reward-value"),
            "reward-card",
            player=EXECUTOR,
        ),
        MoveNode(
            "meld-reward",
            MovementKind.MELD,
            CardSelector.from_variable("reward-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
