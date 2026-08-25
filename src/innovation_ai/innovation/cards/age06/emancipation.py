"""EMANCIPATION - demand one victim hand card into the activator's score pile and,
if transferred, let the victim draw a 6; optionally splay red or purple right.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("emancipation")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "emancipation-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "emancipation-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "emancipation-splay"),
    ),
    (
        SequenceNode(
            "emancipation-demand",
            ("choose-transfer", "transfer-card", "if-transferred"),
        ),
        ChoiceNode(
            "choose-transfer",
            ChoiceKind.HIDDEN_CARD,
            "transferred-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-card",
            MovementKind.TRANSFER,
            CardSelector.from_variable("transferred-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="transferred",
        ),
        ConditionNode("if-transferred", Predicate.truthy("transferred"), "draw-six"),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        SequenceNode("emancipation-splay", ("choose-splay", "splay-right")),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.COLOR,
            "splay-color",
            colors=(Color.RED, Color.PURPLE),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-right",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
