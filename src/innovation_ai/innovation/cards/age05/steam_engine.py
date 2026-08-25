"""STEAM ENGINE - "Draw and tuck two 4, then score your bottom yellow card."

Each draw-and-tuck is an atomic batch. The bottom yellow card is selected only after both tucks,
so a newly tucked yellow card may create the stack that is then scored.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    DrawAndMoveNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    StackPosition,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("steam-engine")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "steam-engine-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "steam-engine-effect"),),
    (
        SequenceNode("steam-engine-effect", ("draw-tuck-two", "score-bottom-yellow")),
        DrawAndMoveNode(
            "draw-tuck-two",
            ValueRef.literal(4),
            ValueRef.literal(2),
            MovementKind.TUCK,
            player=EXECUTOR,
        ),
        MoveNode(
            "score-bottom-yellow",
            MovementKind.SCORE,
            CardSelector.stack(EXECUTOR, color=Color.YELLOW, position=StackPosition.BOTTOM),
            destination_player=EXECUTOR,
        ),
    ),
)
