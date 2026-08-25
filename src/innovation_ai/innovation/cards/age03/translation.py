"""TRANSLATION — optionally meld the entire score pile, then claim World via crowned tops.

The score pile is snapshotted as one indivisible set.  Choosing to meld enters one all-or-none
movement atom; no subset can be selected.  World is claimed through the linked route, whose
universal top-card predicate is intentionally true on an empty board (rules decision 10).
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    AllOrNoneNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    ConditionNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SpecialAchievementId

CARD_ID: Final[CardId] = CardId("translation")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "translation-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "translation-meld"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "translation-world"),
    ),
    (
        SequenceNode("translation-meld", ("snapshot-score", "if-score")),
        LetNode("snapshot-score", "score-cards", cards=CardSelector.score(EXECUTOR)),
        ConditionNode(
            "if-score",
            Predicate.truthy("score-cards"),
            "offer-meld-all",
        ),
        SequenceNode("offer-meld-all", ("choose-meld-all", "if-meld-all")),
        ChoiceNode(
            "choose-meld-all",
            ChoiceKind.BRANCH,
            "meld-all-choice",
            branches=("meld-all",),
            optional=True,
        ),
        ConditionNode(
            "if-meld-all",
            Predicate.truthy("meld-all-choice"),
            "order-and-meld-all",
        ),
        SequenceNode(
            "order-and-meld-all",
            ("order-score-cards", "meld-score-all-or-none"),
        ),
        ChoiceNode(
            "order-score-cards",
            ChoiceKind.ORDER_CARDS,
            "meld-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("score-cards"),
            order_group=OrderGroup.COLOR,
        ),
        AllOrNoneNode(
            "meld-score-all-or-none",
            Predicate.all_match(
                CardSelector.from_variable("score-cards"),
                CardSelector.score(EXECUTOR),
            ),
            "meld-all-score-cards",
        ),
        MoveNode(
            "meld-all-score-cards",
            MovementKind.MELD,
            CardSelector.from_variable("score-cards"),
            destination_player=EXECUTOR,
            order_variable="meld-order",
        ),
        ClaimAchievementNode("translation-world", SpecialAchievementId.WORLD, player=EXECUTOR),
    ),
)
