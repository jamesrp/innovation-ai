"""ECOLOGY - "You may return a card from your hand. If you do, score a card from
your hand and draw two 10."
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
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("ecology")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "ecology-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "ecology-effect"),),
    (
        SequenceNode("ecology-effect", ("choose-return", "return-card", "if-returned")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
            result_variable="returned",
        ),
        ConditionNode("if-returned", Predicate.truthy("returned"), "ecology-reward"),
        SequenceNode("ecology-reward", ("choose-score", "score-card", "draw-two")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "scored-card",
            cards=CardSelector.hand(EXECUTOR),
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-card"),
            destination_player=EXECUTOR,
        ),
        TimesNode("draw-two", ValueRef.literal(2), "draw-ten"),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
    ),
)
