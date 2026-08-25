"""CHEMISTRY - effect 1: "You may splay your blue cards right."
effect 2: "Draw and score a card of value one higher than the highest top card on your board
and then return a card from your score pile."

The requested value is read before the draw-and-score batch, and an empty board therefore requests
age 1. The executor owns the score pile and chooses the exact card to return.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    Extreme,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("chemistry")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "chemistry-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "chemistry-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "chemistry-score-return"),
    ),
    (
        SequenceNode("chemistry-splay", ("choose-blue", "splay-blue")),
        ChoiceNode(
            "choose-blue",
            ChoiceKind.COLOR,
            "blue",
            colors=(Color.BLUE,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-blue",
            EXECUTOR,
            color_variable="blue",
            direction=SplayDirection.RIGHT,
        ),
        SequenceNode(
            "chemistry-score-return",
            ("draw-score-reward", "choose-return", "return-score"),
        ),
        BatchNode("draw-score-reward", ("draw-reward", "score-reward")),
        DrawNode(
            "draw-reward",
            ValueRef.selector_extreme(CardSelector.top_cards(EXECUTOR), Extreme.HIGHEST, offset=1),
            "reward",
            player=EXECUTOR,
        ),
        MoveNode(
            "score-reward",
            MovementKind.SCORE,
            CardSelector.from_variable("reward"),
            destination_player=EXECUTOR,
        ),
        ChoiceNode(
            "choose-return",
            ChoiceKind.HIDDEN_CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "return-score",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
        ),
    ),
)
