"""COMPOSITES - "I demand you transfer all but one card from your hand to my hand!
Also, transfer the highest card from your score pile to my score pile!"

The victim owns both private-zone choices, including a tie for the one highest score card.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("composites")

_ALL_BUT_KEPT: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    exclude_variable="kept-card",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "composites-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "composites-demand"),),
    (
        SequenceNode(
            "composites-demand",
            ("choose-kept", "transfer-hand", "choose-highest-score", "transfer-highest-score"),
        ),
        ChoiceNode(
            "choose-kept",
            ChoiceKind.HIDDEN_CARD,
            "kept-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-hand",
            MovementKind.TRANSFER,
            _ALL_BUT_KEPT,
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
        ),
        ChoiceNode(
            "choose-highest-score",
            ChoiceKind.HIDDEN_CARD,
            "highest-score",
            chooser=EXECUTOR,
            cards=CardSelector.score(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-highest-score",
            MovementKind.TRANSFER,
            CardSelector.from_variable("highest-score"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
        ),
    ),
)
