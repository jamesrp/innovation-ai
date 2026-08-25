"""WRITING - "Draw a 2."

A single deterministic draw uses the normal upward-only supply fallback and terminal handling.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    DrawNode,
    EffectProgram,
    ProgramEffect,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("writing")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "writing-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "draw-two"),),
    (DrawNode("draw-two", ValueRef.literal(2), "drawn"),),
)
