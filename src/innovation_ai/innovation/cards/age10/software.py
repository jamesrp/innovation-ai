"""SOFTWARE - draw and score a 10; then draw and meld two 10s and execute
the second drawn card's non-demand effects without sharing them.
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
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("software")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "software-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "software-score"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "software-nested"),
    ),
    (
        BatchNode("software-score", ("draw-scored-ten", "score-ten")),
        DrawNode("draw-scored-ten", ValueRef.literal(10), "scored-ten", player=EXECUTOR),
        MoveNode(
            "score-ten",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-ten"),
            destination_player=EXECUTOR,
        ),
        SequenceNode(
            "software-nested",
            (
                "draw-and-meld-first-ten",
                "draw-and-meld-second-ten",
                "execute-second-ten",
            ),
        ),
        BatchNode("draw-and-meld-first-ten", ("draw-first-ten", "meld-first-ten")),
        DrawNode("draw-first-ten", ValueRef.literal(10), "first-ten", player=EXECUTOR),
        MoveNode(
            "meld-first-ten",
            MovementKind.MELD,
            CardSelector.from_variable("first-ten"),
            destination_player=EXECUTOR,
        ),
        BatchNode("draw-and-meld-second-ten", ("draw-second-ten", "meld-second-ten")),
        DrawNode("draw-second-ten", ValueRef.literal(10), "second-ten", player=EXECUTOR),
        MoveNode(
            "meld-second-ten",
            MovementKind.MELD,
            CardSelector.from_variable("second-ten"),
            destination_player=EXECUTOR,
        ),
        NestedNode("execute-second-ten", "second-ten"),
    ),
)
