"""RAILROAD - return all hand cards, then draw three 6; optionally splay up one
color currently splayed right.

The return order is owner-chosen where same-age cards share a supply pile. The second printed
choice is represented directly as a semantic color choice.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("railroad")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "railroad-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "railroad-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "railroad-splay"),
    ),
    (
        SequenceNode(
            "railroad-return",
            ("bind-hand", "order-hand", "return-hand", "draw-three-sixes"),
        ),
        LetNode("bind-hand", "returned-hand", cards=CardSelector.hand(EXECUTOR)),
        ChoiceNode(
            "order-hand",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("returned-hand"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-hand",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-hand"),
            order_variable="return-order",
        ),
        TimesNode("draw-three-sixes", ValueRef.literal(3), "draw-six"),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        SequenceNode("railroad-splay", ("choose-right-color", "if-right-color")),
        ChoiceNode(
            "choose-right-color",
            ChoiceKind.COLOR,
            "right-color",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            color_source=ChoiceColorSource.PRESENT_ON_BOARD,
            required_splay=SplayDirection.RIGHT,
            optional=True,
        ),
        ConditionNode(
            "if-right-color",
            Predicate.truthy("right-color"),
            "splay-right-color-up",
        ),
        SplayNode(
            "splay-right-color-up",
            EXECUTOR,
            color_variable="right-color",
            direction=SplayDirection.UP,
            result_variable="splayed",
        ),
    ),
)
