"""DEMOCRACY - grouped hand returns and an explicit per-dogma return-count record.

Each executor chooses the complete subset and meaningful same-age order before one bulk return.
The strict prior maximum is stored directly in this card execution's causal scope; no gameplay or
achievement event arithmetic is involved.
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable, variable_context
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    VariableScope,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("democracy")


def _returned_more_than_every_prior_executor(state: Any, context: Any, registry: Any) -> bool:
    """Compare this executor's explicit count with the persisted prior maximum."""

    del registry
    current = get_effect_variable(state, context, "return-count", 0)
    prior = get_effect_variable(
        state,
        variable_context(context, VariableScope.CARD_EXECUTION),
        "prior-return-count",
        0,
    )
    return (
        isinstance(current, int)
        and not isinstance(current, bool)
        and isinstance(prior, int)
        and not isinstance(prior, bool)
        and current > prior
    )


PREDICATES: Final[dict[str, NamedPredicate]] = {
    "returned-more-than-every-prior-executor": _returned_more_than_every_prior_executor,
}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "democracy-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "democracy-effect"),),
    (
        SequenceNode(
            "democracy-effect",
            (
                "choose-returns",
                "snapshot-return-count",
                "order-returns",
                "return-selected",
                "if-new-record",
            ),
        ),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "selected-returns",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=105,
        ),
        LetNode(
            "snapshot-return-count",
            "return-count",
            value=ValueRef.count("selected-returns"),
        ),
        ChoiceNode(
            "order-returns",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-returns"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-selected",
            MovementKind.RETURN,
            CardSelector.from_variable("selected-returns"),
            order_variable="return-order",
        ),
        ConditionNode(
            "if-new-record",
            Predicate.named("returned-more-than-every-prior-executor"),
            "reward-and-record",
        ),
        SequenceNode(
            "reward-and-record",
            ("draw-and-score-eight", "remember-return-record"),
        ),
        BatchNode("draw-and-score-eight", ("draw-eight", "score-eight")),
        DrawNode("draw-eight", ValueRef.literal(8), "reward-eight", player=EXECUTOR),
        MoveNode(
            "score-eight",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-eight"),
            destination_player=EXECUTOR,
        ),
        LetNode(
            "remember-return-record",
            "prior-return-count",
            value=ValueRef.from_variable("return-count"),
            result_scope=VariableScope.CARD_EXECUTION,
        ),
    ),
)
