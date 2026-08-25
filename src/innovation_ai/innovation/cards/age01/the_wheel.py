"""THE WHEEL - "Draw two 1."

The simplest possible dogma: no choice, no branch. It exists in the slice to pin the trivial
cases - a shared execution that always changes the game and therefore always earns the sharing
bonus, and the legality of taking a Dogma action that can do nothing when the age 1 supply and
every higher pile are empty.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    DrawNode,
    EffectProgram,
    ProgramEffect,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("the-wheel")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "the-wheel-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "draw-two"),),
    (
        TimesNode("draw-two", ValueRef.literal(2), "draw-one"),
        DrawNode("draw-one", ValueRef.literal(1), "drawn"),
    ),
)
