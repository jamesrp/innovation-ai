"""SAILING - "Draw and meld a 1."

The two movements form one atomic draw-and-X instruction, so no intermediate achievement boundary
is visible and the exact card drawn is the card melded.
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
    ProgramEffect,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("sailing")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "sailing-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "draw-and-meld"),),
    (
        BatchNode("draw-and-meld", ("draw-one", "meld-one")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        MoveNode(
            "meld-one",
            MovementKind.MELD,
            CardSelector.from_variable("drawn"),
            destination_player=EXECUTOR,
        ),
    ),
)
