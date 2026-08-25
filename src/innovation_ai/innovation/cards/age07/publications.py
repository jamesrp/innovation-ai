"""PUBLICATIONS - "You may rearrange the order of one color of cards on your board."
Then: "You may splay your yellow or blue cards up."

Only a stack with at least two cards has an order to rearrange. Rearrangement retains the stack's
splay, while the second optional choice includes only existing yellow or blue stacks.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    Predicate,
    ProgramEffect,
    RearrangeNode,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("publications")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "publications-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "publications-rearrange"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "publications-splay"),
    ),
    (
        SequenceNode("publications-rearrange", ("choose-color", "if-color")),
        ChoiceNode(
            "choose-color",
            ChoiceKind.COLOR,
            "rearrange-color",
            colors=tuple(Color),
            optional=True,
            minimum_stack_size=2,
        ),
        ConditionNode(
            "if-color",
            Predicate.truthy("rearrange-color"),
            "rearrange-sequence",
        ),
        SequenceNode("rearrange-sequence", ("choose-order", "apply-order")),
        ChoiceNode(
            "choose-order",
            ChoiceKind.ORDER_CARDS,
            "stack-order",
            chooser=EXECUTOR,
            cards=CardSelector.stack(EXECUTOR, color_variable="rearrange-color"),
        ),
        RearrangeNode(
            "apply-order",
            EXECUTOR,
            "rearrange-color",
            "stack-order",
            result_variable="rearranged",
        ),
        SequenceNode("publications-splay", ("choose-splay-color", "if-splay-color")),
        ChoiceNode(
            "choose-splay-color",
            ChoiceKind.COLOR,
            "splay-color",
            colors=(Color.YELLOW, Color.BLUE),
            optional=True,
            minimum_stack_size=1,
        ),
        ConditionNode(
            "if-splay-color",
            Predicate.truthy("splay-color"),
            "splay-up",
        ),
        SplayNode(
            "splay-up",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.UP,
            result_variable="splayed",
        ),
    ),
)
