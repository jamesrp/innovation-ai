"""CLASSIFICATION - reveal only a hand card's colour, take every opponent hand card
of that colour, then meld every matching card from the executor's hand.

The executor's exact initial card choice stays private: no ``RevealNode`` is used because the card
prints "reveal the color", not "reveal a card".  A colour binding drives relational selectors;
all matching transfers and melds are mandatory.  Only the order of same-colour melds is chosen,
because that order determines the resulting top card.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ALL_OTHER_PLAYERS,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("classification")

_CHOSEN_CARD: Final = CardSelector.from_variable("colour-card")
_MATCHING_OPPONENT_CARDS: Final = CardSelector(
    CardSelectorKind.HAND,
    ALL_OTHER_PLAYERS,
    relation=SelectorRelation(SelectorRelationKind.SAME_COLOR_AS_ANY, _CHOSEN_CARD),
)
_MATCHING_EXECUTOR_CARDS: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    relation=SelectorRelation(SelectorRelationKind.SAME_COLOR_AS_ANY, _CHOSEN_CARD),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "classification-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "classification-effect"),),
    (
        SequenceNode(
            "classification-effect",
            (
                "choose-colour-card",
                "bind-revealed-colour",
                "take-matching-cards",
                "snapshot-melds",
                "order-melds",
                "meld-matching-cards",
            ),
        ),
        ChoiceNode(
            "choose-colour-card",
            ChoiceKind.CARD,
            "colour-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
        ),
        LetNode("bind-revealed-colour", "revealed-colour", color_of="colour-card"),
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
