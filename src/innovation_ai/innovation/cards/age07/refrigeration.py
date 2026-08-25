"""REFRIGERATION - demand: "I demand you return half (rounded down) of the cards in
your hand!" Then: "You may score a card from your hand."

``TimesNode`` snapshots ``floor(hand size / 2)`` before the first return. The victim owns every
private-card choice and therefore selects the exact identities and their return order.
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
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("refrigeration")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "refrigeration-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "refrigeration-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "refrigeration-score"),
    ),
    (
        TimesNode(
            "refrigeration-demand",
            ValueRef.count_selector(CardSelector.hand(EXECUTOR), per=2),
            "return-one-sequence",
        ),
        SequenceNode("return-one-sequence", ("choose-return", "return-card")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.HIDDEN_CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
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
