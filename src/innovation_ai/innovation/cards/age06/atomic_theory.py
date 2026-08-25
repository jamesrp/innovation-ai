"""ATOMIC THEORY - optionally splay blue right, then draw and meld a 7."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("atomic-theory")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "atomic-theory-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "atomic-theory-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "draw-and-meld-seven"),
    ),
    (
        SequenceNode("atomic-theory-splay", ("choose-blue", "splay-blue")),
        ChoiceNode(
            "choose-blue",
            ChoiceKind.COLOR,
            "blue-splay",
            colors=(Color.BLUE,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-blue",
            EXECUTOR,
            color_variable="blue-splay",
            direction=SplayDirection.RIGHT,
        ),
        BatchNode("draw-and-meld-seven", ("draw-seven", "meld-seven")),
        DrawNode("draw-seven", ValueRef.literal(7), "drawn-seven", player=EXECUTOR),
        MoveNode(
            "meld-seven",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-seven"),
            destination_player=EXECUTOR,
        ),
    ),
)
