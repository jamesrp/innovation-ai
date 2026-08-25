"""COMPUTERS - optionally splay red or green up; draw and meld a 10, then execute
that card's non-demand effects without a new sharing pass.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
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
    NestedNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("computers")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "computers-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "computers-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "computers-nested"),
    ),
    (
        SequenceNode("computers-splay", ("choose-splay", "if-splay")),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.COLOR,
            "splay-color",
            target_player=EXECUTOR,
            colors=(Color.RED, Color.GREEN),
            optional=True,
            minimum_stack_size=1,
        ),
        ConditionNode("if-splay", Predicate.truthy("splay-color"), "splay-up"),
        SplayNode(
            "splay-up",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.UP,
        ),
        SequenceNode("computers-nested", ("draw-and-meld-ten", "execute-ten")),
        BatchNode("draw-and-meld-ten", ("draw-ten", "meld-ten")),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
        MoveNode(
            "meld-ten",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-ten"),
            destination_player=EXECUTOR,
        ),
        NestedNode("execute-ten", "drawn-ten"),
    ),
)
