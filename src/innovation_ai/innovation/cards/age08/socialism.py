"""SOCIALISM - optionally tuck the complete hand as an all-or-none operation; if
that hand contained purple, take every tied-lowest card from the opponent's hand.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    AllOrNoneNode,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    Extreme,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("socialism")

_ORIGINAL_HAND: Final = CardSelector.from_variable("original-hand")
_ORIGINAL_PURPLES: Final = CardSelector(
    CardSelectorKind.VARIABLE,
    variable="original-hand",
    colors=(Color.PURPLE,),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "socialism-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "socialism-effect"),),
    (
        SequenceNode("socialism-effect", ("choose-all-or-none", "if-tuck-all")),
        ChoiceNode(
            "choose-all-or-none",
            ChoiceKind.BRANCH,
            "tuck-choice",
            chooser=EXECUTOR,
            branches=("tuck-all",),
            optional=True,
        ),
        ConditionNode(
            "if-tuck-all",
            Predicate.truthy("tuck-choice"),
            "tuck-complete-hand",
        ),
        SequenceNode(
            "tuck-complete-hand",
            ("snapshot-hand", "order-hand", "all-or-none-tuck", "if-purple"),
        ),
        LetNode("snapshot-hand", "original-hand", cards=CardSelector.hand(EXECUTOR)),
        ChoiceNode(
            "order-hand",
            ChoiceKind.ORDER_CARDS,
            "tuck-order",
            chooser=EXECUTOR,
            cards=_ORIGINAL_HAND,
            order_group=OrderGroup.COLOR,
        ),
        AllOrNoneNode(
            "all-or-none-tuck",
            Predicate.all_match(_ORIGINAL_HAND, CardSelector.hand(EXECUTOR)),
            "tuck-hand-batch",
        ),
        BatchNode("tuck-hand-batch", ("tuck-all-cards",)),
        MoveNode(
            "tuck-all-cards",
            MovementKind.TUCK,
            _ORIGINAL_HAND,
            destination_player=EXECUTOR,
            order_variable="tuck-order",
        ),
        ConditionNode(
            "if-purple",
            Predicate.non_empty(_ORIGINAL_PURPLES),
            "take-lowest-opponent-cards",
        ),
        MoveNode(
            "take-lowest-opponent-cards",
            MovementKind.TRANSFER,
            CardSelector.hand(OPPONENT, extreme=Extreme.LOWEST),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.HAND,
        ),
    ),
)
