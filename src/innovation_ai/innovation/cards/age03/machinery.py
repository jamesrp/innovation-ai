"""MACHINERY — mandatory whole-hand/highest-card exchange, castle score, optional red splay."""

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
    ExchangeNode,
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("machinery")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "machinery-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "machinery-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "machinery-score-splay"),
    ),
    (
        ExchangeNode(
            "machinery-demand",
            CardSelector.hand(EXECUTOR),
            CardSelector.hand(ACTIVATOR, extreme=Extreme.HIGHEST),
        ),
        SequenceNode(
            "machinery-score-splay",
            ("choose-castle", "score-castle", "if-red-stack"),
        ),
        ChoiceNode(
            "choose-castle",
            ChoiceKind.CARD,
            "castle-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR, icon=Icon.CASTLE),
        ),
        MoveNode(
            "score-castle",
            MovementKind.SCORE,
            CardSelector.from_variable("castle-card"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "if-red-stack",
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
