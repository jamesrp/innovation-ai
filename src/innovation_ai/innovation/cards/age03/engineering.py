"""ENGINEERING — demand all castle tops, then optionally splay red left."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
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
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("engineering")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "engineering-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "engineering-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "engineering-splay"),
    ),
    (
        MoveNode(
            "engineering-demand",
            MovementKind.TRANSFER,
            CardSelector.top_cards(EXECUTOR, icon=Icon.CASTLE),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
        ),
        ConditionNode(
            "engineering-splay",
            Predicate.non_empty(CardSelector.stack(EXECUTOR, color=Color.RED)),
            "offer-red-splay",
        ),
        SequenceNode("offer-red-splay", ("choose-red-splay", "if-red-splay")),
        ChoiceNode(
            "choose-red-splay",
            ChoiceKind.BRANCH,
            "red-splay",
            branches=("splay-left",),
            optional=True,
        ),
        ConditionNode("if-red-splay", Predicate.truthy("red-splay"), "splay-red-left"),
        SplayNode(
            "splay-red-left",
            EXECUTOR,
            color=Color.RED,
            direction=SplayDirection.LEFT,
        ),
    ),
)
