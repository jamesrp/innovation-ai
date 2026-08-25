"""FERMENTING - "Draw a 2 for every color on your board with one or more leaf."

Each qualifying color is counted once from its visible icons.  ``TimesNode`` snapshots that count
before the first draw, so newly drawn cards cannot alter the remaining quantity.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    DrawNode,
    EffectProgram,
    ProgramEffect,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("fermenting")

_COLORS_WITH_LEAVES: Final[ValueRef] = ValueRef(
    ValueRefKind.COLORS_WITH_ICON,
    icon=Icon.LEAF,
    player=EXECUTOR,
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "fermenting-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "fermenting-effect"),),
    (
        TimesNode(
            "fermenting-effect",
            _COLORS_WITH_LEAVES,
            "draw-two",
            maximum_iterations=5,
        ),
        DrawNode("draw-two", ValueRef.literal(2), "drawn-two", player=EXECUTOR),
    ),
)
