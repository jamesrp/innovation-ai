"""MONOTHEISM - demand: "I demand you transfer a top card on your board of different
color from any card on my board to my score pile! If you do, draw and tuck a 1!" Then: "Draw and
tuck a 1."

The relational selector compares against every card on the activator's board, including covered
cards.  The conditional victim reward and the independent second effect are atomic draw-and-tuck
instructions.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("monotheism")

_TRANSFERABLE_TOPS: Final[CardSelector] = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    relation=SelectorRelation(
        SelectorRelationKind.DIFFERENT_COLOR_FROM_ALL,
        CardSelector.board(ACTIVATOR),
    ),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "monotheism-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "monotheism-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "monotheism-draw-tuck"),
    ),
    (
        SequenceNode(
            "monotheism-demand",
            ("choose-top", "transfer-top", "if-transferred"),
        ),
        ChoiceNode(
            "choose-top",
            ChoiceKind.CARD,
            "transferred-top",
            chooser=EXECUTOR,
            cards=_TRANSFERABLE_TOPS,
        ),
        MoveNode(
            "transfer-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("transferred-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="did-transfer",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("did-transfer"),
            "victim-draw-tuck",
        ),
        BatchNode("victim-draw-tuck", ("victim-draw-one", "victim-tuck-one")),
        DrawNode("victim-draw-one", ValueRef.literal(1), "victim-one", player=EXECUTOR),
        MoveNode(
            "victim-tuck-one",
            MovementKind.TUCK,
            CardSelector.from_variable("victim-one"),
            destination_player=EXECUTOR,
        ),
        BatchNode("monotheism-draw-tuck", ("draw-one", "tuck-one")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn-one", player=EXECUTOR),
        MoveNode(
            "tuck-one",
            MovementKind.TUCK,
            CardSelector.from_variable("drawn-one"),
            destination_player=EXECUTOR,
        ),
    ),
)
