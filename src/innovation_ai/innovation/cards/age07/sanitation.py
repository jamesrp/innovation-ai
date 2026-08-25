"""SANITATION - "I demand you exchange the two highest cards in your hand with the
lowest card in my hand!"

The victim chooses each of their highest private cards, recomputing after excluding the first.
The activator owns the other hidden hand and disambiguates tied lowest identities. Named selector
predicates then feed those exact identities to one atomic exchange leaf.
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
    EffectProgram,
    ExchangeNode,
    Extreme,
    ExtremeScope,
    NamedPredicate,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("sanitation")


def _candidate_is_selected(state: Any, context: Any, registry: Any) -> bool:
    """Whether a candidate is one of the exact hidden identities selected for exchange."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    selected = {
        get_effect_variable(state, context, variable)
        for variable in ("highest-one", "highest-two", "lowest-card")
    }
    return isinstance(raw_candidate, str) and raw_candidate in selected


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-selected": _candidate_is_selected,
}

_SECOND_HIGHEST: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    extreme=Extreme.HIGHEST,
    extreme_scope=ExtremeScope.ONE_TIED,
    exclude_variable="highest-one",
)

_SELECTED_VICTIM_CARDS: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    predicate="candidate-is-selected",
)

_SELECTED_ACTIVATOR_CARDS: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    ACTIVATOR,
    predicate="candidate-is-selected",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "sanitation-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "sanitation-demand"),),
    (
        SequenceNode(
            "sanitation-demand",
            ("choose-highest-one", "choose-highest-two", "choose-lowest", "exchange-selected"),
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
            cards=_SECOND_HIGHEST,
            owner=EXECUTOR,
        ),
        ChoiceNode(
            "choose-lowest",
            ChoiceKind.HIDDEN_CARD,
            "lowest-card",
            chooser=ACTIVATOR,
            cards=CardSelector.hand(
                ACTIVATOR,
                extreme=Extreme.LOWEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=ACTIVATOR,
        ),
        ExchangeNode(
            "exchange-selected",
            _SELECTED_VICTIM_CARDS,
            _SELECTED_ACTIVATOR_CARDS,
            result_variable="exchanged",
        ),
    ),
)
