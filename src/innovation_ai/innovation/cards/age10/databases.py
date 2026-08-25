"""DATABASES - choose half the score pile, then return the complete group atomically.

The rounded-up count is snapshotted before any choice. Exact hidden identities are collected
against the unchanged score pile, same-age order is chosen separately, and one bulk movement
commits the return.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    CollectNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    Rounding,
    SequenceNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("databases")

_UNSELECTED_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    EXECUTOR,
    exclude_variable="selected-score-cards",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "databases-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "databases-demand"),),
    (
        SequenceNode(
            "databases-demand",
            ("collect-score-cards", "order-score-cards", "return-score-cards"),
        ),
        TimesNode(
            "collect-score-cards",
            ValueRef(
                ValueRefKind.COUNT_SELECTOR,
                selector=CardSelector.score(EXECUTOR),
                per=2,
                rounding=Rounding.CEIL,
            ),
            "collect-one-score",
        ),
        SequenceNode("collect-one-score", ("choose-score-card", "remember-score-card")),
        ChoiceNode(
            "choose-score-card",
            ChoiceKind.HIDDEN_CARD,
            "returned-score-card",
            chooser=EXECUTOR,
            cards=_UNSELECTED_SCORES,
            owner=EXECUTOR,
        ),
        CollectNode("remember-score-card", "returned-score-card", "selected-score-cards"),
        ChoiceNode(
            "order-score-cards",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-score-cards"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-score-cards",
            MovementKind.RETURN,
            CardSelector.from_variable("selected-score-cards"),
            order_variable="return-order",
        ),
    ),
)
