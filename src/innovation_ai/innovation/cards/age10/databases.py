"""DATABASES - "I demand you return half (rounded up) of the cards in your score pile!"

The return count is snapshotted before the first return. The victim owns the hidden score pile and
therefore chooses each exact card identity.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    Rounding,
    SequenceNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("databases")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "databases-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "databases-demand"),),
    (
        TimesNode(
            "databases-demand",
            ValueRef(
                ValueRefKind.COUNT_SELECTOR,
                selector=CardSelector.score(EXECUTOR),
                per=2,
                rounding=Rounding.CEIL,
            ),
            "return-one-sequence",
        ),
        SequenceNode("return-one-sequence", ("choose-score-card", "return-score-card")),
        ChoiceNode(
            "choose-score-card",
            ChoiceKind.HIDDEN_CARD,
            "returned-score-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "return-score-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-score-card"),
        ),
    ),
)
