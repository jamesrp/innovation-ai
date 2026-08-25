"""MEDICINE — exchange one tied-highest victim score card with one tied-lowest owner card.

Rules decision 13 assigns each hidden tie to the score pile's owner.  The pure named selector
connects those exact choices to the atomic exchange without exposing either private pile to the
other chooser.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.effects.model import EffectContext
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

CARD_ID: Final[CardId] = CardId("medicine")


def _candidate_is_chosen_exchange_card(
    state: Any, context: EffectContext, registry: CardRegistry
) -> bool:
    """Match the reserved selector candidate to either owner's exact tied-card choice."""

    del registry
    candidate_key = f"{context.scope}:_candidate"
    chosen_keys = {
        f"{context.scope}:victim-highest",
        f"{context.scope}:owner-lowest",
    }
    candidate = next(
        (variable.value for variable in state.effect_variables if variable.name == candidate_key),
        None,
    )
    chosen = {
        variable.value
        for variable in state.effect_variables
        if variable.name in chosen_keys and isinstance(variable.value, str)
    }
    return isinstance(candidate, str) and candidate in chosen


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-chosen-exchange-card": _candidate_is_chosen_exchange_card,
}

_CHOSEN_VICTIM_SCORE: Final = CardSelector(
    CardSelectorKind.SCORE,
    EXECUTOR,
    predicate="candidate-is-chosen-exchange-card",
)
_CHOSEN_OWNER_SCORE: Final = CardSelector(
    CardSelectorKind.SCORE,
    ACTIVATOR,
    predicate="candidate-is-chosen-exchange-card",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "medicine-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "medicine-demand"),),
    (
        SequenceNode(
            "medicine-demand",
            ("choose-victim-highest", "choose-owner-lowest", "exchange-score-cards"),
        ),
        ChoiceNode(
            "choose-victim-highest",
            ChoiceKind.HIDDEN_CARD,
            "victim-highest",
            chooser=EXECUTOR,
            cards=CardSelector.score(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=EXECUTOR,
        ),
        ChoiceNode(
            "choose-owner-lowest",
            ChoiceKind.HIDDEN_CARD,
            "owner-lowest",
            chooser=ACTIVATOR,
            cards=CardSelector.score(
                ACTIVATOR,
                extreme=Extreme.LOWEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=ACTIVATOR,
        ),
        ExchangeNode("exchange-score-cards", _CHOSEN_VICTIM_SCORE, _CHOSEN_OWNER_SCORE),
    ),
)
