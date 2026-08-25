"""AGRICULTURE - "You may return a card from your hand. If you do, draw and score a card
of value one higher than the card you returned."

The returned card's value is bound before it leaves the hand.  The reward's draw-and-score is a
single atomic instruction, while declining or having an empty hand leaves the effect unchanged.
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

CARD_ID: Final[CardId] = CardId("agriculture")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "agriculture-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "agriculture-effect"),),
    (
        SequenceNode("agriculture-effect", ("choose-return", "if-chosen")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        ConditionNode("if-chosen", Predicate.truthy("returned-card"), "return-and-reward"),
        SequenceNode(
            "return-and-reward",
            ("bind-reward-value", "return-card", "draw-and-score"),
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
            result_variable="did-return",
        ),
        BatchNode("draw-and-score", ("draw-reward", "score-reward")),
        DrawNode(
            "draw-reward",
            ValueRef.from_variable("reward-value"),
            "reward-card",
            player=EXECUTOR,
        ),
        MoveNode(
            "score-reward",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
