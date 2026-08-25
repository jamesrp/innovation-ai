"""ROBOTICS - score the top green card, then draw and meld a 10 and execute
that card's non-demand effects without sharing them.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    NestedNode,
    ProgramEffect,
    SequenceNode,
    StackPosition,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("robotics")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "robotics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "robotics-effect"),),
    (
        SequenceNode(
            "robotics-effect",
            ("score-green-top", "draw-and-meld-ten", "execute-ten"),
        ),
        MoveNode(
            "score-green-top",
            MovementKind.SCORE,
            CardSelector.stack(EXECUTOR, color=Color.GREEN, position=StackPosition.TOP),
            destination_player=EXECUTOR,
        ),
        BatchNode("draw-and-meld-ten", ("draw-ten", "meld-ten")),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
        MoveNode(
            "meld-ten",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-ten"),
            destination_player=EXECUTOR,
        ),
        NestedNode("execute-ten", "drawn-ten"),
    ),
)
