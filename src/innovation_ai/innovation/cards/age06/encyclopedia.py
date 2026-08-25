"""ENCYCLOPEDIA - optionally meld the complete set of highest score cards.

The only initial alternatives are "meld all" and decline.  The highest set is snapshotted, its
same-colour movement order is chosen separately, and one all-or-none node guards the single bulk
meld instruction so no partial highest-card meld is observable.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    AllOrNoneNode,
    CardSelector,
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
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("encyclopedia")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "encyclopedia-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "encyclopedia-effect"),),
    (
        ConditionNode(
            "encyclopedia-effect",
            Predicate.non_empty(CardSelector.score(EXECUTOR)),
            "offer-meld",
        ),
        SequenceNode("offer-meld", ("choose-all", "if-all")),
        ChoiceNode(
            "choose-all",
            ChoiceKind.BRANCH,
            "meld-choice",
            branches=("meld-all-highest",),
            optional=True,
        ),
        ConditionNode("if-all", Predicate.truthy("meld-choice"), "meld-all-sequence"),
        SequenceNode(
            "meld-all-sequence",
            ("snapshot-highest", "order-highest", "meld-all-or-none"),
        ),
        LetNode(
            "snapshot-highest",
            "highest-cards",
            cards=CardSelector.score(EXECUTOR, extreme=Extreme.HIGHEST),
        ),
        ChoiceNode(
            "order-highest",
            ChoiceKind.ORDER_CARDS,
            "meld-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("highest-cards"),
            order_group=OrderGroup.COLOR,
        ),
        AllOrNoneNode(
            "meld-all-or-none",
            Predicate.all_match(
                CardSelector.from_variable("highest-cards"),
                CardSelector.score(EXECUTOR),
            ),
            "meld-highest",
        ),
        MoveNode(
            "meld-highest",
            MovementKind.MELD,
            CardSelector.from_variable("highest-cards"),
            destination_player=EXECUTOR,
            order_variable="meld-order",
            moved_variable="melded-highest",
        ),
    ),
)
