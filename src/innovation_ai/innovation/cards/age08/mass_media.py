"""MASS MEDIA - optionally return a hand card, choose any value, and return every
card of that value from both score piles; then optionally splay purple up.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ALL_PLAYERS,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("mass-media")

_ALL_CHOSEN_VALUE_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    ALL_PLAYERS,
    value_expr=ValueRef.from_variable("chosen-value"),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mass-media-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "mass-media-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "mass-media-splay"),
    ),
    (
        SequenceNode(
            "mass-media-return",
            ("choose-hand-return", "return-hand-card", "if-returned"),
        ),
        ChoiceNode(
            "choose-hand-return",
            ChoiceKind.CARD,
            "returned-hand-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "return-hand-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-hand-card"),
            result_variable="did-return",
        ),
        ConditionNode(
            "if-returned",
            Predicate.truthy("did-return"),
            "choose-and-return-scores",
        ),
        SequenceNode(
            "choose-and-return-scores",
            ("choose-value", "return-all-chosen-value-scores"),
        ),
        ChoiceNode(
            "choose-value",
            ChoiceKind.VALUE,
            "chosen-value",
            chooser=EXECUTOR,
            values=tuple(range(1, 11)),
        ),
        MoveNode(
            "return-all-chosen-value-scores",
            MovementKind.RETURN,
            _ALL_CHOSEN_VALUE_SCORES,
        ),
        SequenceNode("mass-media-splay", ("choose-purple", "splay-purple")),
        ChoiceNode(
            "choose-purple",
            ChoiceKind.COLOR,
            "purple-color",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            colors=(Color.PURPLE,),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-purple",
            EXECUTOR,
            color_variable="purple-color",
            direction=SplayDirection.UP,
        ),
    ),
)
