"""PHILOSOPHY - effect 1: "You may splay left any one color of your cards."
Effect 2: "You may score a card from your hand."

Every color present on the executor's board is a legal optional splay choice, including singleton
or already-left stacks under rules decision 15.  Scoring is a separate optional exact-card choice.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("philosophy")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "philosophy-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "philosophy-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "philosophy-score"),
    ),
    (
        SequenceNode("philosophy-splay", ("choose-color", "splay-left")),
        ChoiceNode(
            "choose-color",
            ChoiceKind.COLOR,
            "splay-color",
            chooser=EXECUTOR,
            optional=True,
            color_source=ChoiceColorSource.PRESENT_ON_BOARD,
        ),
        SplayNode(
            "splay-left",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.LEFT,
        ),
        SequenceNode("philosophy-score", ("choose-score", "score-card")),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "scored-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            optional=True,
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("scored-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
