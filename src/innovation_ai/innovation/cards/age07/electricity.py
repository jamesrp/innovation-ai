"""ELECTRICITY - "Return all your top cards without a factory, then draw an 8 for each
card you returned."

The complete top-card set and its count are bound before movement. Return ordering is a distinct
owner choice only for cards sharing an age pile; the reward quantity remains the original count.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("electricity")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "electricity-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "electricity-effect"),),
    (
        SequenceNode(
            "electricity-effect",
            ("bind-returned", "bind-count", "order-returned", "return-tops", "draw-eights"),
        ),
        LetNode(
            "bind-returned",
            "returned-tops",
            cards=CardSelector.top_cards(EXECUTOR, without_icon=Icon.FACTORY),
        ),
        LetNode("bind-count", "return-count", value=ValueRef.count("returned-tops")),
        ChoiceNode(
            "order-returned",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("returned-tops"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-tops",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-tops"),
            order_variable="return-order",
        ),
        TimesNode("draw-eights", ValueRef.from_variable("return-count"), "draw-eight"),
        DrawNode("draw-eight", ValueRef.literal(8), "drawn-eight", player=EXECUTOR),
    ),
)
