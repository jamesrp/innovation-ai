"""PRINTING PRESS - optionally return a score card, draw above purple, then splay blue."""

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
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    StackPosition,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("printing-press")

_TOP_PURPLE: Final[CardSelector] = CardSelector.stack(
    EXECUTOR,
    color=Color.PURPLE,
    position=StackPosition.TOP,
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "printing-press-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "printing-press-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "printing-press-splay"),
    ),
    (
        SequenceNode("printing-press-return", ("choose-score", "if-score")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
            optional=True,
        ),
        ConditionNode("if-score", Predicate.truthy("score-card"), "return-and-draw"),
        SequenceNode("return-and-draw", ("return-score", "draw-above-purple")),
        MoveNode(
            "return-score",
            MovementKind.RETURN,
            CardSelector.from_variable("score-card"),
        ),
        DrawNode(
            "draw-above-purple",
            ValueRef.selector_extreme(_TOP_PURPLE, Extreme.HIGHEST, offset=2),
            "drawn-card",
            player=EXECUTOR,
        ),
        SequenceNode("printing-press-splay", ("choose-blue", "splay-blue")),
        ChoiceNode(
            "choose-blue",
            ChoiceKind.COLOR,
            "blue",
            chooser=EXECUTOR,
            colors=(Color.BLUE,),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-blue",
            EXECUTOR,
            color_variable="blue",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
