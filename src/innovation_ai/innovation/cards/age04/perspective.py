"""PERSPECTIVE - optionally return a hand card, then score once per two bulbs."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("perspective")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "perspective-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "perspective-effect"),),
    (
        SequenceNode("perspective-effect", ("choose-return", "if-returned-card")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        ConditionNode(
            "if-returned-card",
            Predicate.truthy("returned-card"),
            "return-and-score",
        ),
        SequenceNode("return-and-score", ("return-card", "score-for-bulbs")),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
        ),
        TimesNode(
            "score-for-bulbs",
            ValueRef.icon_count(Icon.LIGHTBULB, EXECUTOR, per=2),
            "score-one-sequence",
        ),
        SequenceNode("score-one-sequence", ("choose-score", "score-card")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "scored-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
