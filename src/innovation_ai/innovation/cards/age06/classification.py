"""CLASSIFICATION - publicly reveal a semantic hand colour, then transfer and meld it.

The first choice enumerates distinct colours present in the executor's hand, never exact private
card identities. A transient public colour marker and reveal event remain visible while the
matching transfer and meld instruction resolves.
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    ALL_OTHER_PLAYERS,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    RevealColorNode,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("classification")


def _matches_revealed_color(state: Any, context: Any, registry: Any) -> bool:
    candidate = get_effect_variable(state, context, "_candidate")
    chosen = get_effect_variable(state, context, "revealed-colour")
    return (
        isinstance(candidate, str)
        and isinstance(chosen, str)
        and registry.card(CardId(candidate)).color.value == chosen
    )


PREDICATES: Final = {"matches-revealed-color": _matches_revealed_color}

_MATCHING_OPPONENT_CARDS: Final = CardSelector(
    CardSelectorKind.HAND,
    ALL_OTHER_PLAYERS,
    predicate="matches-revealed-color",
)
_MATCHING_EXECUTOR_CARDS: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    predicate="matches-revealed-color",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "classification-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "classification-effect"),),
    (
        SequenceNode(
            "classification-effect",
            (
                "choose-colour",
                "reveal-colour",
                "take-matching-cards",
                "snapshot-melds",
                "order-melds",
                "meld-matching-cards",
            ),
        ),
        ChoiceNode(
            "choose-colour",
            ChoiceKind.COLOR,
            "revealed-colour",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            color_source=ChoiceColorSource.PRESENT_IN_HAND,
        ),
        RevealColorNode("reveal-colour", "revealed-colour"),
        MoveNode(
            "take-matching-cards",
            MovementKind.TRANSFER,
            _MATCHING_OPPONENT_CARDS,
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.HAND,
            moved_variable="taken-cards",
        ),
        LetNode("snapshot-melds", "matching-cards", cards=_MATCHING_EXECUTOR_CARDS),
        ChoiceNode(
            "order-melds",
            ChoiceKind.ORDER_CARDS,
            "meld-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("matching-cards"),
            order_group=OrderGroup.COLOR,
        ),
        MoveNode(
            "meld-matching-cards",
            MovementKind.MELD,
            CardSelector.from_variable("matching-cards"),
            destination_player=EXECUTOR,
            order_variable="meld-order",
            moved_variable="melded-cards",
        ),
    ),
)
