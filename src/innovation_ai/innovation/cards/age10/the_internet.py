"""THE INTERNET - optionally splay green up, draw and score a 10, then draw and
meld one 10 for every two clocks visible when the third instruction begins.
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
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("the-internet")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "the-internet-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "the-internet-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "the-internet-score"),
        ProgramEffect(DogmaEffectId(CARD_ID, 3), False, "the-internet-melds"),
    ),
    (
        SequenceNode("the-internet-splay", ("choose-green-splay", "if-green-splay")),
        ChoiceNode(
            "choose-green-splay",
            ChoiceKind.COLOR,
            "green-splay",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            colors=(Color.GREEN,),
            optional=True,
            minimum_stack_size=1,
        ),
        ConditionNode(
            "if-green-splay",
            Predicate.truthy("green-splay"),
            "splay-green-up",
        ),
        SplayNode(
            "splay-green-up",
            EXECUTOR,
            color=Color.GREEN,
            direction=SplayDirection.UP,
        ),
        BatchNode("the-internet-score", ("draw-scored-ten", "score-ten")),
        DrawNode("draw-scored-ten", ValueRef.literal(10), "scored-ten", player=EXECUTOR),
        MoveNode(
            "score-ten",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-ten"),
            destination_player=EXECUTOR,
        ),
        TimesNode(
            "the-internet-melds",
            ValueRef.icon_count(Icon.CLOCK, EXECUTOR, per=2),
            "draw-and-meld-ten",
        ),
        BatchNode("draw-and-meld-ten", ("draw-melded-ten", "meld-ten")),
        DrawNode("draw-melded-ten", ValueRef.literal(10), "melded-ten", player=EXECUTOR),
        MoveNode(
            "meld-ten",
            MovementKind.MELD,
            CardSelector.from_variable("melded-ten"),
            destination_player=EXECUTOR,
        ),
    ),
)
