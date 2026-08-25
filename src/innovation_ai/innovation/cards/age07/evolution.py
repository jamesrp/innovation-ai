"""EVOLUTION - "You may choose to either draw and score an 8 and then return a card
from your score pile, or draw a card of value one higher than the highest card in your score
pile."

One optional branch choice represents the complete printed alternative. The return happens after
the age-8 card is scored, so that new card is itself a legal return candidate.
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
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("evolution")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "evolution-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "evolution-effect"),),
    (
        SequenceNode("evolution-effect", ("choose-branch", "if-score-eight")),
        ChoiceNode(
            "choose-branch",
            ChoiceKind.BRANCH,
            "evolution-branch",
            branches=("score-eight", "draw-above-score"),
            optional=True,
        ),
        ConditionNode(
            "if-score-eight",
            Predicate.equals("evolution-branch", "score-eight"),
            "score-eight-sequence",
            "if-draw-above",
        ),
        SequenceNode(
            "score-eight-sequence",
            ("draw-eight", "score-eight", "choose-score-return", "return-score-card"),
        ),
        DrawNode("draw-eight", ValueRef.literal(8), "drawn-eight", player=EXECUTOR),
        MoveNode(
            "score-eight",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-eight"),
            destination_player=EXECUTOR,
        ),
        ChoiceNode(
            "choose-score-return",
            ChoiceKind.CARD,
            "score-return",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
        ),
        MoveNode(
            "return-score-card",
            MovementKind.RETURN,
            CardSelector.from_variable("score-return"),
        ),
        ConditionNode(
            "if-draw-above",
            Predicate.equals("evolution-branch", "draw-above-score"),
            "draw-above-score",
        ),
        DrawNode(
            "draw-above-score",
            ValueRef.selector_extreme(CardSelector.score(EXECUTOR), Extreme.HIGHEST, offset=1),
            "drawn-above",
            player=EXECUTOR,
        ),
    ),
)
