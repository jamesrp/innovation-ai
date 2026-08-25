"""ASTRONOMY - effect 1: "Draw and reveal a 6. If the card is green or blue, meld it
and repeat this dogma effect."
effect 2: "If all non-purple top cards on your board are value 6 or higher, claim the
Universe achievement."

The repeat is scoped to the first printed effect. The linked Universe route deliberately uses
vacuous truth when the executor has no non-purple top cards (rules decision 10).
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ClaimAchievementNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RepeatNode,
    RevealNode,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SpecialAchievementId

CARD_ID: Final[CardId] = CardId("astronomy")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "astronomy-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "astronomy-repeat"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "claim-universe"),
    ),
    (
        RepeatNode(
            "astronomy-repeat",
            "astronomy-body",
            Predicate.card_color_in("drawn", (Color.GREEN, Color.BLUE)),
            maximum_iterations=105,
        ),
        SequenceNode("astronomy-body", ("draw-and-reveal", "if-green-or-blue")),
        BatchNode("draw-and-reveal", ("draw-six", "reveal-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn", player=EXECUTOR),
        RevealNode("reveal-six", CardSelector.from_variable("drawn")),
        ConditionNode(
            "if-green-or-blue",
            Predicate.card_color_in("drawn", (Color.GREEN, Color.BLUE)),
            "meld-drawn",
        ),
        MoveNode(
            "meld-drawn",
            MovementKind.MELD,
            CardSelector.from_variable("drawn"),
            destination_player=EXECUTOR,
        ),
        ClaimAchievementNode(
            "claim-universe",
            SpecialAchievementId.UNIVERSE,
            player=EXECUTOR,
        ),
    ),
)
