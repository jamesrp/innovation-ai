"""SPECIALIZATION - reveal a hand card and take every opponent top card of that
color; optionally splay yellow or blue up.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ALL_OTHER_PLAYERS,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    SplayNode,
    StackPosition,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("specialization")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "specialization-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "specialization-take"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "specialization-splay"),
    ),
    (
        SequenceNode(
            "specialization-take",
            ("choose-reveal", "reveal-card", "bind-color", "take-opponent-tops"),
        ),
        ChoiceNode(
            "choose-reveal",
            ChoiceKind.CARD,
            "revealed-card",
            cards=CardSelector.hand(EXECUTOR),
        ),
        RevealNode("reveal-card", CardSelector.from_variable("revealed-card")),
        LetNode("bind-color", "revealed-color", color_of="revealed-card"),
        MoveNode(
            "take-opponent-tops",
            MovementKind.TRANSFER,
            CardSelector.stack(
                ALL_OTHER_PLAYERS,
                color_variable="revealed-color",
                position=StackPosition.TOP,
            ),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.HAND,
        ),
        SequenceNode("specialization-splay", ("choose-splay", "if-splay")),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.COLOR,
            "splay-color",
            target_player=EXECUTOR,
            colors=(Color.YELLOW, Color.BLUE),
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
    ),
)
