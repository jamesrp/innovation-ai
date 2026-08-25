"""STEM CELLS - optionally score the complete hand as one all-or-none instruction.

No subset is ever offered: accepting the branch scores the snapshotted hand in one atomic movement
leaf, while declining leaves every card in place.
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
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("stem-cells")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "stem-cells-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "stem-cells-effect"),),
    (
        SequenceNode("stem-cells-effect", ("snapshot-hand", "if-hand")),
        LetNode("snapshot-hand", "hand-cards", cards=CardSelector.hand(EXECUTOR)),
        ConditionNode("if-hand", Predicate.truthy("hand-cards"), "offer-score-all"),
        SequenceNode("offer-score-all", ("choose-score-all", "if-score-all")),
        ChoiceNode(
            "choose-score-all",
            ChoiceKind.BRANCH,
            "score-all-choice",
            chooser=EXECUTOR,
            branches=("score-all",),
            optional=True,
        ),
        ConditionNode(
            "if-score-all",
            Predicate.truthy("score-all-choice"),
            "score-all-or-none",
        ),
        AllOrNoneNode(
            "score-all-or-none",
            Predicate.all_match(
                CardSelector.from_variable("hand-cards"),
                CardSelector.hand(EXECUTOR),
            ),
            "score-complete-hand",
        ),
        MoveNode(
            "score-complete-hand",
            MovementKind.SCORE,
            CardSelector.from_variable("hand-cards"),
            destination_player=EXECUTOR,
        ),
    ),
)
