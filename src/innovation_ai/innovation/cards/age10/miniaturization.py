"""MINIATURIZATION - optionally return a hand card; returning a 10 draws one 10
for every distinct value represented in the executor's score pile.
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
    Predicate,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("miniaturization")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "miniaturization-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "miniaturization-effect"),),
    (
        SequenceNode(
            "miniaturization-effect",
            ("choose-return", "return-card", "if-returned-ten"),
        ),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
        ),
        ConditionNode(
            "if-returned-ten",
            Predicate.count(
                ValueRef.card_value("returned-card"),
                Cmp.EQ,
                ValueRef.literal(10),
            ),
            "draw-for-distinct-values",
        ),
        SequenceNode(
            "draw-for-distinct-values",
            ("snapshot-score", "draw-tens"),
        ),
        LetNode("snapshot-score", "score-cards", cards=CardSelector.score(EXECUTOR)),
        TimesNode(
            "draw-tens",
            ValueRef(ValueRefKind.DISTINCT_VALUES, variable="score-cards"),
            "draw-ten",
        ),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
    ),
)
