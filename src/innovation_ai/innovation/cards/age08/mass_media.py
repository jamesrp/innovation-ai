"""MASS MEDIA - optionally return a hand card, choose any value, and return every
card of that value from both score piles; then optionally splay purple up.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("mass-media")

_EXECUTOR_CHOSEN_VALUE_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    EXECUTOR,
    value_expr=ValueRef.from_variable("chosen-value"),
)
_OPPONENT_CHOSEN_VALUE_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    OPPONENT,
    value_expr=ValueRef.from_variable("chosen-value"),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mass-media-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "mass-media-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "mass-media-splay"),
    ),
    (
        SequenceNode(
            "mass-media-return",
            ("choose-hand-return", "return-hand-card", "if-returned"),
        ),
        ChoiceNode(
            "choose-hand-return",
            ChoiceKind.CARD,
            "returned-hand-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "return-hand-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-hand-card"),
            result_variable="did-return",
        ),
        ConditionNode(
            "if-returned",
            Predicate.truthy("did-return"),
            "choose-and-return-scores",
        ),
        SequenceNode(
            "choose-and-return-scores",
            (
                "choose-value",
                "order-executor-scores",
                "order-opponent-scores",
                "return-chosen-value-scores",
            ),
        ),
        ChoiceNode(
            "choose-value",
            ChoiceKind.VALUE,
            "chosen-value",
            chooser=EXECUTOR,
            values=tuple(range(1, 11)),
        ),
        ChoiceNode(
            "order-executor-scores",
            ChoiceKind.ORDER_CARDS,
            "executor-return-order",
            chooser=EXECUTOR,
            cards=_EXECUTOR_CHOSEN_VALUE_SCORES,
            order_group=OrderGroup.AGE,
        ),
        ChoiceNode(
            "order-opponent-scores",
            ChoiceKind.ORDER_CARDS,
            "opponent-return-order",
            chooser=OPPONENT,
            cards=_OPPONENT_CHOSEN_VALUE_SCORES,
            order_group=OrderGroup.AGE,
        ),
        BatchNode(
            "return-chosen-value-scores",
            ("return-executor-scores", "return-opponent-scores"),
        ),
        MoveNode(
            "return-executor-scores",
            MovementKind.RETURN,
            _EXECUTOR_CHOSEN_VALUE_SCORES,
            order_variable="executor-return-order",
        ),
        MoveNode(
            "return-opponent-scores",
            MovementKind.RETURN,
            _OPPONENT_CHOSEN_VALUE_SCORES,
            order_variable="opponent-return-order",
        ),
        SequenceNode("mass-media-splay", ("choose-purple", "splay-purple")),
        ChoiceNode(
            "choose-purple",
            ChoiceKind.COLOR,
            "purple-color",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            colors=(Color.PURPLE,),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-purple",
            EXECUTOR,
            color_variable="purple-color",
            direction=SplayDirection.UP,
        ),
    ),
)
