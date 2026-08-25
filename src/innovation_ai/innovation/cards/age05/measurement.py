"""MEASUREMENT - "You may reveal and return a card from your hand. If you do, splay
that color of your cards right, and draw a card of value equal to the number of cards of that
color on your board."

The hand-card choice is unrestricted. In particular, a card matching a singleton stack remains a
legal choice: the mandatory re-splay is a no-op, but the player still draws the specified 1 under
the official Measurement clarification and rules decision 15.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
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
    RevealNode,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("measurement")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "measurement-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "measurement-effect"),),
    (
        SequenceNode("measurement-effect", ("choose-card", "if-chosen")),
        ChoiceNode(
            "choose-card",
            ChoiceKind.CARD,
            "chosen-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        ConditionNode("if-chosen", Predicate.truthy("chosen-card"), "return-splay-draw"),
        SequenceNode(
            "return-splay-draw",
            ("bind-color", "reveal-card", "return-card", "splay-color", "bind-count", "draw"),
        ),
        LetNode("bind-color", "chosen-color", color_of="chosen-card"),
        RevealNode("reveal-card", CardSelector.from_variable("chosen-card")),
        MoveNode("return-card", MovementKind.RETURN, CardSelector.from_variable("chosen-card")),
        SplayNode(
            "splay-color",
            EXECUTOR,
            color_variable="chosen-color",
            direction=SplayDirection.RIGHT,
        ),
        LetNode(
            "bind-count",
            "color-count",
            value=ValueRef.count_selector(
                CardSelector.stack(EXECUTOR, color_variable="chosen-color")
            ),
        ),
        DrawNode(
            "draw",
            ValueRef.from_variable("color-count"),
            "drawn",
            player=EXECUTOR,
        ),
    ),
)
