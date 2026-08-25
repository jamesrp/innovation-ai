"""ENTERPRISE - demand a non-purple crown top, reward the victim, then offer green splay."""

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

CARD_ID: Final[CardId] = CardId("enterprise")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "enterprise-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "enterprise-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "enterprise-splay"),
    ),
    (
        SequenceNode(
            "enterprise-demand",
            ("choose-crown-top", "transfer-crown-top", "if-transferred"),
        ),
        ChoiceNode(
            "choose-crown-top",
            ChoiceKind.CARD,
            "crown-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(
                EXECUTOR,
                icon=Icon.CROWN,
                exclude_colors=(Color.PURPLE,),
            ),
        ),
        MoveNode(
            "transfer-crown-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("crown-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
            result_variable="transferred",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("transferred"),
            "draw-meld-four",
        ),
        BatchNode("draw-meld-four", ("draw-four", "meld-four")),
        DrawNode("draw-four", ValueRef.literal(4), "drawn-four", player=EXECUTOR),
        MoveNode(
            "meld-four",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-four"),
            destination_player=EXECUTOR,
        ),
        SequenceNode("enterprise-splay", ("choose-green", "splay-green")),
        ChoiceNode(
            "choose-green",
            ChoiceKind.COLOR,
            "green",
            chooser=EXECUTOR,
            colors=(Color.GREEN,),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-green",
            EXECUTOR,
            color_variable="green",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
