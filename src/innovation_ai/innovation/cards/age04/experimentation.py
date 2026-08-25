"""EXPERIMENTATION - draw and meld a 5 as one atomic instruction."""

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

CARD_ID: Final[CardId] = CardId("experimentation")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "experimentation-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "draw-meld-five"),),
    (
        BatchNode("draw-meld-five", ("draw-five", "meld-five")),
        DrawNode("draw-five", ValueRef.literal(5), "drawn-five", player=EXECUTOR),
        MoveNode(
            "meld-five",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-five"),
            destination_player=EXECUTOR,
        ),
    ),
)
