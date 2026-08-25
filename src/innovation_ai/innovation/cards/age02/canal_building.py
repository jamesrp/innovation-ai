"""CANAL BUILDING - "You may exchange all the highest cards in your hand with all the
highest cards in your score pile."

The optional branch is offered whenever at least one side is nonempty.  The exchange itself is a
single atomic leaf and includes every card tied for the highest value on either side.
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
    ExchangeNode,
    Extreme,
    Predicate,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("canal-building")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "canal-building-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "canal-building-effect"),),
    (
        ConditionNode(
            "canal-building-effect",
            Predicate.non_empty(CardSelector.hand(EXECUTOR)),
            "offer-exchange",
            "if-score-nonempty",
        ),
        ConditionNode(
            "if-score-nonempty",
            Predicate.non_empty(CardSelector.score(EXECUTOR)),
            "offer-exchange",
        ),
        SequenceNode("offer-exchange", ("choose-exchange", "if-exchange")),
        ChoiceNode(
            "choose-exchange",
            ChoiceKind.BRANCH,
            "exchange-choice",
            branches=("exchange-highest",),
            optional=True,
        ),
        ConditionNode(
            "if-exchange",
            Predicate.truthy("exchange-choice"),
            "exchange-highest-cards",
        ),
        ExchangeNode(
            "exchange-highest-cards",
            CardSelector.hand(EXECUTOR, extreme=Extreme.HIGHEST),
            CardSelector.score(EXECUTOR, extreme=Extreme.HIGHEST),
        ),
    ),
)
