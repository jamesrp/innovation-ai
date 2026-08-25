"""SATELLITES - return the hand and draw three 8; optionally splay purple up; then
meld a hand card and execute only that card's non-demand effects.
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
    MovementKind,
    MoveNode,
    NestedNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("satellites")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "satellites-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "satellites-reset"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "satellites-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 3), False, "satellites-nested"),
    ),
    (
        SequenceNode("satellites-reset", ("order-hand", "return-hand", "draw-three")),
        ChoiceNode(
            "order-hand",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            cards=CardSelector.hand(EXECUTOR),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-hand",
            MovementKind.RETURN,
            CardSelector.hand(EXECUTOR),
            order_variable="return-order",
        ),
        TimesNode("draw-three", ValueRef.literal(3), "draw-eight"),
        DrawNode("draw-eight", ValueRef.literal(8), "drawn-eight", player=EXECUTOR),
        SequenceNode("satellites-splay", ("choose-purple", "if-purple")),
        ChoiceNode(
            "choose-purple",
            ChoiceKind.COLOR,
            "purple-splay",
            target_player=EXECUTOR,
            colors=(Color.PURPLE,),
            optional=True,
            minimum_stack_size=1,
        ),
        ConditionNode("if-purple", Predicate.truthy("purple-splay"), "splay-purple"),
        SplayNode(
            "splay-purple",
            EXECUTOR,
            color=Color.PURPLE,
            direction=SplayDirection.UP,
        ),
        SequenceNode("satellites-nested", ("choose-meld", "meld-card", "execute-card")),
        ChoiceNode(
            "choose-meld",
            ChoiceKind.CARD,
            "melded-card",
            cards=CardSelector.hand(EXECUTOR),
        ),
        MoveNode(
            "meld-card",
            MovementKind.MELD,
            CardSelector.from_variable("melded-card"),
            destination_player=EXECUTOR,
        ),
        NestedNode("execute-card", "melded-card"),
    ),
)
