"""PAPER — optionally splay green or blue left, then draw per left-splayed color."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("paper")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "paper-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "paper-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "paper-draw"),
    ),
    (
        SequenceNode("paper-splay", ("choose-splay-color", "splay-left")),
        ChoiceNode(
            "choose-splay-color",
            ChoiceKind.COLOR,
            "splay-color",
            chooser=EXECUTOR,
            colors=(Color.GREEN, Color.BLUE),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-left",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.LEFT,
        ),
        TimesNode(
            "paper-draw",
            ValueRef(
                ValueRefKind.COLORS_SPLAYED,
                player=EXECUTOR,
                direction=SplayDirection.LEFT,
            ),
            "draw-four",
        ),
        DrawNode("draw-four", ValueRef.literal(4), "drawn", player=EXECUTOR),
    ),
)
