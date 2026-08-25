"""ANATOMY - return a victim-chosen score card, then a matching-value board top.

The first choice is deliberately unrestricted.  The official clarification allows the victim to
choose a score card whose value matches no top card unless every score-card choice would cause the
follow-up return.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("anatomy")

_MATCHING_TOP: Final[CardSelector] = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    value_expr=ValueRef.from_variable("returned-value"),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "anatomy-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "anatomy-demand"),),
    (
        SequenceNode("anatomy-demand", ("choose-score", "if-score")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.HIDDEN_CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
            owner=EXECUTOR,
        ),
        ConditionNode("if-score", Predicate.truthy("score-card"), "return-sequence"),
        SequenceNode(
            "return-sequence",
            ("bind-value", "return-score", "choose-matching-top", "return-matching-top"),
        ),
        LetNode(
            "bind-value",
            "returned-value",
            value=ValueRef.card_value("score-card"),
        ),
        MoveNode(
            "return-score",
            MovementKind.RETURN,
            CardSelector.from_variable("score-card"),
        ),
        ChoiceNode(
            "choose-matching-top",
            ChoiceKind.CARD,
            "matching-top",
            chooser=EXECUTOR,
            cards=_MATCHING_TOP,
        ),
        MoveNode(
            "return-matching-top",
            MovementKind.RETURN,
            CardSelector.from_variable("matching-top"),
        ),
    ),
)
