"""COAL - effect 1: "Draw and tuck a 5."
effect 2: "You may splay your red cards right."
effect 3: "You may score any one of your top cards. If you do, also score the card beneath it."

The beneath card is snapshotted before either score movement, then both cards are scored in one
batch. Choosing a singleton top therefore still scores that top and simply has no beneath card.
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
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    StackPosition,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("coal")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "coal-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "draw-tuck-five"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "coal-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 3), False, "coal-score"),
    ),
    (
        BatchNode("draw-tuck-five", ("draw-five", "tuck-five")),
        DrawNode("draw-five", ValueRef.literal(5), "drawn-five", player=EXECUTOR),
        MoveNode(
            "tuck-five",
            MovementKind.TUCK,
            CardSelector.from_variable("drawn-five"),
            destination_player=EXECUTOR,
        ),
        SequenceNode("coal-splay", ("choose-red", "splay-red")),
        ChoiceNode(
            "choose-red",
            ChoiceKind.COLOR,
            "red",
            colors=(Color.RED,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-red",
            EXECUTOR,
            color_variable="red",
            direction=SplayDirection.RIGHT,
        ),
        SequenceNode("coal-score", ("choose-top", "if-chosen")),
        ChoiceNode(
            "choose-top",
            ChoiceKind.CARD,
            "chosen-top",
            cards=CardSelector.top_cards(EXECUTOR),
            optional=True,
        ),
        ConditionNode("if-chosen", Predicate.truthy("chosen-top"), "score-top-and-beneath"),
        SequenceNode(
            "score-top-and-beneath",
            ("bind-beneath", "score-pair"),
        ),
        LetNode(
            "bind-beneath",
            "beneath",
            cards=CardSelector.board(
                EXECUTOR,
                position=StackPosition.BENEATH_VARIABLE,
                position_variable="chosen-top",
            ),
        ),
        BatchNode("score-pair", ("score-top", "score-beneath")),
        MoveNode(
            "score-top",
            MovementKind.SCORE,
            CardSelector.from_variable("chosen-top"),
            destination_player=EXECUTOR,
        ),
        MoveNode(
            "score-beneath",
            MovementKind.SCORE,
            CardSelector.from_variable("beneath"),
            destination_player=EXECUTOR,
        ),
    ),
)
