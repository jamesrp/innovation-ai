"""CORPORATIONS - demand a non-green factory top into the activator's score pile,
reward a successful transfer with a drawn-and-melded 8, then draw and meld an 8.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    BatchNode,
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
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("corporations")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "corporations-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "corporations-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "corporations-meld"),
    ),
    (
        SequenceNode(
            "corporations-demand",
            ("choose-factory-top", "transfer-factory-top", "if-transferred"),
        ),
        ChoiceNode(
            "choose-factory-top",
            ChoiceKind.CARD,
            "factory-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(
                EXECUTOR,
                icon=Icon.FACTORY,
                exclude_colors=(Color.GREEN,),
            ),
        ),
        MoveNode(
            "transfer-factory-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("factory-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="did-transfer",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("did-transfer"),
            "victim-draw-and-meld",
        ),
        BatchNode("victim-draw-and-meld", ("victim-draw-eight", "victim-meld-eight")),
        DrawNode("victim-draw-eight", ValueRef.literal(8), "victim-eight", player=EXECUTOR),
        MoveNode(
            "victim-meld-eight",
            MovementKind.MELD,
            CardSelector.from_variable("victim-eight"),
            destination_player=EXECUTOR,
        ),
        BatchNode("corporations-meld", ("executor-draw-eight", "executor-meld-eight")),
        DrawNode("executor-draw-eight", ValueRef.literal(8), "executor-eight", player=EXECUTOR),
        MoveNode(
            "executor-meld-eight",
            MovementKind.MELD,
            CardSelector.from_variable("executor-eight"),
            destination_player=EXECUTOR,
        ),
    ),
)
