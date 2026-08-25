"""NAVIGATION - demand one victim-chosen value-2 or value-3 score card."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    NamedPredicate,
    ProgramEffect,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("navigation")


def _candidate_is_value_two_or_three(state: Any, context: Any, registry: Any) -> bool:
    """Whether the reserved private-score candidate has one of Navigation's values."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    return isinstance(raw_candidate, str) and registry.card(CardId(raw_candidate)).age in {2, 3}


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-value-two-or-three": _candidate_is_value_two_or_three,
}

_ELIGIBLE_SCORE: Final[CardSelector] = CardSelector(
    CardSelectorKind.SCORE,
    EXECUTOR,
    predicate="candidate-is-value-two-or-three",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "navigation-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "navigation-demand"),),
    (
        SequenceNode("navigation-demand", ("choose-score", "transfer-score")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.HIDDEN_CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=_ELIGIBLE_SCORE,
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-score",
            MovementKind.TRANSFER,
            CardSelector.from_variable("score-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
        ),
    ),
)
