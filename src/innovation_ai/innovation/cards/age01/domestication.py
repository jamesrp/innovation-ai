"""DOMESTICATION - "Meld the lowest card in your hand. Draw a 1."

The hand owner chooses among tied lowest cards.  An empty hand simply skips the meld and still
performs the independent draw.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("domestication")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "domestication-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "domestication-effect"),),
    (
        SequenceNode("domestication-effect", ("choose-lowest", "meld-lowest", "draw-one")),
        ChoiceNode(
            "choose-lowest",
            ChoiceKind.HIDDEN_CARD,
            "lowest-card",
            chooser=EXECUTOR,
            owner=EXECUTOR,
            cards=CardSelector.hand(
                EXECUTOR,
                extreme=Extreme.LOWEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
        ),
        MoveNode(
            "meld-lowest",
            MovementKind.MELD,
            CardSelector.from_variable("lowest-card"),
            destination_player=EXECUTOR,
        ),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
    ),
)
