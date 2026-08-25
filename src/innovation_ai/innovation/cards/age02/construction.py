"""CONSTRUCTION - demand: "I demand you transfer two cards from your hand to my hand!
Draw a 2!" Then: "If you are the only player with five top cards, claim the Empire achievement."

The victim selects an exact two-card subset, with mandatory partial execution when fewer cards
exist.  The independent draw always follows.  Effect two uses the frozen linked Empire route.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SpecialAchievementId

CARD_ID: Final[CardId] = CardId("construction")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "construction-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "construction-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "construction-empire"),
    ),
    (
        SequenceNode(
            "construction-demand",
            ("choose-two", "transfer-two", "draw-two"),
        ),
        ChoiceNode(
            "choose-two",
            ChoiceKind.BOUNDED_CARDS,
            "transferred-cards",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=2,
            maximum=2,
        ),
        MoveNode(
            "transfer-two",
            MovementKind.TRANSFER,
            CardSelector.from_variable("transferred-cards"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
        ),
        DrawNode("draw-two", ValueRef.literal(2), "drawn-two", player=EXECUTOR),
        ClaimAchievementNode(
            "construction-empire",
            SpecialAchievementId.EMPIRE,
            player=EXECUTOR,
        ),
    ),
)
