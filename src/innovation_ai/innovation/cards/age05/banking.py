"""BANKING - effect 1: "I demand you transfer a top non-green card with a factory from
your board to my board. If you do, draw and score a 5!"
effect 2: "You may splay your green cards right."

The demand victim chooses the public qualifying top card. The conditional draw-and-score belongs
to that victim ("you") and is one atomic batch. A fixed-colour splay is offered only when that
colour exists, while singleton and already-right stacks remain legal no-op choices (decision 15).
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
    SplayNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("banking")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "banking-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "banking-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "banking-splay"),
    ),
    (
        SequenceNode(
            "banking-demand",
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
            destination_zone=ZoneKind.BOARD,
            result_variable="transferred",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("transferred"),
            "draw-score-five",
        ),
        BatchNode("draw-score-five", ("draw-five", "score-five")),
        DrawNode("draw-five", ValueRef.literal(5), "drawn-five", player=EXECUTOR),
        MoveNode(
            "score-five",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-five"),
            destination_player=EXECUTOR,
        ),
        SequenceNode("banking-splay", ("choose-green", "splay-green")),
        ChoiceNode(
            "choose-green",
            ChoiceKind.COLOR,
            "green",
            colors=(Color.GREEN,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-green",
            EXECUTOR,
            color_variable="green",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
