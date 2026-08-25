"""MACHINE TOOLS - draw and score a card equal to the highest score-card value.

An empty score pile requests value zero, which the shared draw rule starts at age 1 (rules
decision 1).  Draw and score form one atomic instruction.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    DrawNode,
    EffectProgram,
    Extreme,
    MovementKind,
    MoveNode,
    ProgramEffect,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("machine-tools")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "machine-tools-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "machine-tools-effect"),),
    (
        BatchNode("machine-tools-effect", ("draw-score-card", "score-card")),
        DrawNode(
            "draw-score-card",
            ValueRef.selector_extreme(CardSelector.score(EXECUTOR), Extreme.HIGHEST),
            "drawn-card",
            player=EXECUTOR,
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
