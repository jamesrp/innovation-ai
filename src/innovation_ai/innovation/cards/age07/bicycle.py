"""BICYCLE - "You may exchange all the cards in your hand with all the cards in your
score pile. If you exchange one, you must exchange them all."

The optional branch never exposes a partial selection. The complete hand/score exchange is one
atomic leaf behind an all-or-none guard, including when either zone is empty.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    AllOrNoneNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    EffectProgram,
    ExchangeNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("bicycle")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "bicycle-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "bicycle-effect"),),
    (
        ConditionNode(
            "bicycle-effect",
            Predicate.non_empty(CardSelector.hand(EXECUTOR)),
            "offer-exchange-sequence",
            "if-score-nonempty",
        ),
        ConditionNode(
            "if-score-nonempty",
            Predicate.non_empty(CardSelector.score(EXECUTOR)),
            "offer-exchange-sequence",
        ),
        SequenceNode("offer-exchange-sequence", ("offer-exchange", "if-exchange")),
        ChoiceNode(
            "offer-exchange",
            ChoiceKind.BRANCH,
            "exchange-choice",
            branches=("exchange",),
            optional=True,
        ),
        ConditionNode(
            "if-exchange",
            Predicate.truthy("exchange-choice"),
            "exchange-all-or-none",
        ),
        # Exchanging complete zones is always feasible, even when one side is empty. Keeping the
        # operation behind AllOrNone makes the printed prohibition on a partial exchange explicit.
        AllOrNoneNode(
            "exchange-all-or-none",
            Predicate.count(ValueRef.literal(1), Cmp.EQ, ValueRef.literal(1)),
            "exchange-zones",
        ),
        ExchangeNode(
            "exchange-zones",
            CardSelector.hand(EXECUTOR),
            CardSelector.score(EXECUTOR),
            result_variable="exchanged",
        ),
    ),
)
