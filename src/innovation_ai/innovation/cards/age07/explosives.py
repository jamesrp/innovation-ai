"""EXPLOSIVES - "I demand you transfer the three highest cards from your hand to my
hand! If you transferred any, and then have no cards in hand, draw a 7!"

The three private identities are selected against the still-unmoved hand, recomputing the highest
value after excluding prior selections. One transfer leaf then moves the complete set atomically.
Ties belong to the demand victim. The initial hand count records whether any transfer could occur
before the post-demand empty-hand test.
"""

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
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("explosives")


def _candidate_is_unselected(state: Any, context: Any, registry: Any) -> bool:
    """Whether the current private-hand candidate has not already been selected."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    selected = {
        get_effect_variable(state, context, variable)
        for variable in ("highest-one", "highest-two", "highest-three")
    }
    return isinstance(raw_candidate, str) and raw_candidate not in selected


def _candidate_is_selected(state: Any, context: Any, registry: Any) -> bool:
    """Whether the current private-hand candidate is one of the chosen highest cards."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    selected = {
        get_effect_variable(state, context, variable)
        for variable in ("highest-one", "highest-two", "highest-three")
    }
    return isinstance(raw_candidate, str) and raw_candidate in selected


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-unselected": _candidate_is_unselected,
    "candidate-is-selected": _candidate_is_selected,
}

_REMAINING_HIGHEST: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    extreme=Extreme.HIGHEST,
    extreme_scope=ExtremeScope.ONE_TIED,
    predicate="candidate-is-unselected",
)

_SELECTED_HIGHEST: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    predicate="candidate-is-selected",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "explosives-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "explosives-demand"),),
    (
        SequenceNode(
            "explosives-demand",
            (
                "bind-initial-count",
                "choose-highest-one",
                "choose-highest-two",
                "choose-highest-three",
                "transfer-highest",
                "if-any-transferred",
            ),
        ),
        LetNode(
            "bind-initial-count",
            "initial-hand-count",
            value=ValueRef.count_selector(CardSelector.hand(EXECUTOR)),
        ),
        ChoiceNode(
            "choose-highest-one",
            ChoiceKind.HIDDEN_CARD,
            "highest-one",
            chooser=EXECUTOR,
            cards=CardSelector.hand(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=EXECUTOR,
        ),
        ChoiceNode(
            "choose-highest-two",
            ChoiceKind.HIDDEN_CARD,
            "highest-two",
            chooser=EXECUTOR,
            cards=_REMAINING_HIGHEST,
            owner=EXECUTOR,
        ),
        ChoiceNode(
            "choose-highest-three",
            ChoiceKind.HIDDEN_CARD,
            "highest-three",
            chooser=EXECUTOR,
            cards=_REMAINING_HIGHEST,
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-highest",
            MovementKind.TRANSFER,
            _SELECTED_HIGHEST,
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
        ),
        ConditionNode(
            "if-any-transferred",
            Predicate.count(
                ValueRef.from_variable("initial-hand-count"),
                Cmp.GT,
                ValueRef.literal(0),
            ),
            "if-hand-empty",
        ),
        ConditionNode(
            "if-hand-empty",
            Predicate.count(
                ValueRef.count_selector(CardSelector.hand(EXECUTOR)),
                Cmp.EQ,
                ValueRef.literal(0),
            ),
            "draw-seven",
        ),
        DrawNode("draw-seven", ValueRef.literal(7), "drawn-seven", player=EXECUTOR),
    ),
)
