"""REFRIGERATION - demand return half the victim's hand, rounded down; then the
executor may score a hand card.

The mandatory subset and its same-age return order are fully chosen before one grouped return.
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
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("refrigeration")

_UNSELECTED_HAND: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    exclude_variable="selected-returns",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "refrigeration-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "refrigeration-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "refrigeration-score"),
    ),
    (
        SequenceNode(
            "refrigeration-demand",
            ("collect-returns", "order-returns", "return-selected"),
        ),
        TimesNode(
            "collect-returns",
            ValueRef.count_selector(CardSelector.hand(EXECUTOR), per=2),
            "collect-one-return",
        ),
        SequenceNode("collect-one-return", ("choose-return", "remember-return")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.HIDDEN_CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=_UNSELECTED_HAND,
            owner=EXECUTOR,
        ),
        CollectNode("remember-return", "returned-card", "selected-returns"),
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
        SequenceNode("refrigeration-score", ("choose-score", "score-card")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "scored-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
