"""FEUDALISM — demand a castle card, unsplay its color, then offer a left splay."""

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
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("feudalism")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "feudalism-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "feudalism-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "feudalism-splay"),
    ),
    (
        SequenceNode(
            "feudalism-demand",
            ("choose-castle", "if-castle"),
        ),
        ChoiceNode(
            "choose-castle",
            ChoiceKind.HIDDEN_CARD,
            "castle-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR, icon=Icon.CASTLE),
            owner=EXECUTOR,
        ),
        ConditionNode("if-castle", Predicate.truthy("castle-card"), "transfer-and-unsplay"),
        SequenceNode(
            "transfer-and-unsplay",
            ("remember-color", "transfer-castle", "if-transferred"),
        ),
        LetNode("remember-color", "castle-color", color_of="castle-card"),
        MoveNode(
            "transfer-castle",
            MovementKind.TRANSFER,
            CardSelector.from_variable("castle-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
            result_variable="did-transfer",
        ),
        ConditionNode("if-transferred", Predicate.truthy("did-transfer"), "unsplay-color"),
        SplayNode(
            "unsplay-color",
            EXECUTOR,
            color_variable="castle-color",
            direction=SplayDirection.NONE,
        ),
        SequenceNode("feudalism-splay", ("choose-splay-color", "splay-chosen-color")),
        ChoiceNode(
            "choose-splay-color",
            ChoiceKind.COLOR,
            "splay-color",
            chooser=EXECUTOR,
            colors=(Color.YELLOW, Color.PURPLE),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-chosen-color",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.LEFT,
        ),
    ),
)
