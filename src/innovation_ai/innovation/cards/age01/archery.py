"""ARCHERY - "I demand you draw a 1, then transfer the highest card in your hand to my hand!"

This is the slice's demand card. It pins the pronoun mapping ("your hand" is the executing
opponent's, "my hand" is the activator's), immunity at equal featured-icon counts, and the hidden
tie case: the highest card in a hand the demander cannot see.

Rules decision 13 governs that tie. The demand victim owns and may inspect their own hand, so
they are the chooser among equally-highest cards even though the demander only knows the value.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
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
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("archery")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "archery-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "archery-demand"),),
    (
        SequenceNode("archery-demand", ("draw-one", "choose-highest", "transfer-highest")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        # "the highest card" is one card, and a tie is broken by the hand's owner.
        ChoiceNode(
            "choose-highest",
            ChoiceKind.HIDDEN_CARD,
            "highest-card",
            chooser=EXECUTOR,
            cards=CardSelector.hand(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-highest",
            MovementKind.TRANSFER,
            CardSelector.from_variable("highest-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
        ),
    ),
)
