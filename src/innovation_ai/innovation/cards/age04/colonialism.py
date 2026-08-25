"""COLONIALISM - draw and tuck a 3, repeating while the drawn card has a crown."""

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
    Predicate,
    ProgramEffect,
    RepeatNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("colonialism")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "colonialism-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "colonialism-repeat"),),
    (
        RepeatNode(
            "colonialism-repeat",
            "draw-tuck-three",
            Predicate.card_has_icon("drawn-three", Icon.CROWN),
            maximum_iterations=105,
        ),
        BatchNode("draw-tuck-three", ("draw-three", "tuck-three")),
        DrawNode("draw-three", ValueRef.literal(3), "drawn-three", player=EXECUTOR),
        MoveNode(
            "tuck-three",
            MovementKind.TUCK,
            CardSelector.from_variable("drawn-three"),
            destination_player=EXECUTOR,
        ),
    ),
)
