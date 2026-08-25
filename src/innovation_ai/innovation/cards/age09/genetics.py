"""GENETICS - "Draw and meld a 10. Score all cards beneath it."

The destination stack is snapshotted before the meld; those are exactly the cards beneath the
new 10 afterward, including every covered card rather than only the immediately adjacent one.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("genetics")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "genetics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "genetics-effect"),),
    (
        SequenceNode(
            "genetics-effect",
            ("draw-and-meld-ten", "score-beneath"),
        ),
        BatchNode(
            "draw-and-meld-ten",
            ("draw-ten", "bind-color", "snapshot-stack", "meld-ten"),
        ),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
        LetNode("bind-color", "drawn-color", color_of="drawn-ten"),
        LetNode(
            "snapshot-stack",
            "cards-beneath",
            cards=CardSelector.stack(EXECUTOR, color_variable="drawn-color"),
        ),
        MoveNode(
            "meld-ten",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-ten"),
            destination_player=EXECUTOR,
        ),
        MoveNode(
            "score-beneath",
            MovementKind.SCORE,
            CardSelector.from_variable("cards-beneath"),
            destination_player=EXECUTOR,
        ),
    ),
)
