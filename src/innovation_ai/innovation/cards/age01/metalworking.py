"""METALWORKING - "Draw and reveal a 1. If it has a castle, score it and repeat this dogma
effect. Otherwise, keep it."

Metalworking is the slice's repeat/reveal/branch card. It pins:

* a physical reveal marker (decision 18) that is visible in the observation while the card is
  face up and is cleared the moment the card is scored or kept;
* "repeat this dogma effect", which per decision 17 starts a new execution and reevaluates from
  the then-current state;
* a deterministic iteration ceiling, so a pathological supply cannot loop forever.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ConditionNode,
    DrawNode,
    EffectProgram,
    KeepNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RepeatNode,
    RevealNode,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("metalworking")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "metalworking-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "metalworking-repeat"),),
    (
        # There are only fifteen age 1 cards, so sixteen iterations cannot be reached legally and
        # a breach is a genuine engine defect rather than a long game.
        RepeatNode(
            "metalworking-repeat",
            "metalworking-body",
            Predicate.card_has_icon("drawn", Icon.CASTLE),
            maximum_iterations=16,
        ),
        SequenceNode("metalworking-body", ("draw-one", "reveal-one", "castle-branch")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        RevealNode("reveal-one", CardSelector.from_variable("drawn")),
        ConditionNode(
            "castle-branch",
            Predicate.card_has_icon("drawn", Icon.CASTLE),
            "score-drawn",
            "keep-drawn",
        ),
        MoveNode(
            "score-drawn",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn"),
            destination_player=EXECUTOR,
        ),
        KeepNode("keep-drawn", CardSelector.from_variable("drawn")),
    ),
)
