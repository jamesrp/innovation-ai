"""REFORMATION - optionally tuck up to one card per two leaves, then splay right."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
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
    StopNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("reformation")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "reformation-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "reformation-tuck"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "reformation-splay"),
    ),
    (
        TimesNode(
            "reformation-tuck",
            ValueRef.icon_count(Icon.LEAF, EXECUTOR, per=2),
            "offer-one-tuck",
        ),
        SequenceNode("offer-one-tuck", ("choose-tuck", "if-tuck")),
        ChoiceNode(
            "choose-tuck",
            ChoiceKind.CARD,
            "tucked-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        ConditionNode(
            "if-tuck",
            Predicate.truthy("tucked-card"),
            "tuck-card",
            "stop-tucking",
        ),
        MoveNode(
            "tuck-card",
            MovementKind.TUCK,
            CardSelector.from_variable("tucked-card"),
            destination_player=EXECUTOR,
        ),
        StopNode("stop-tucking"),
        SequenceNode("reformation-splay", ("choose-splay", "splay-right")),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.COLOR,
            "splay-color",
            chooser=EXECUTOR,
            colors=(Color.YELLOW, Color.PURPLE),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-right",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
