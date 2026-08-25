"""CALENDAR - "If you have more cards in your score pile than in your hand, draw two 3."

The comparison is evaluated when the effect reaches its branch.  A successful branch performs
both draws sequentially, preserving upward supply fallback and immediate terminal handling.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Predicate,
    ProgramEffect,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("calendar")

_MORE_SCORED_THAN_HELD: Final[Predicate] = Predicate.count(
    ValueRef.count_selector(CardSelector.score(EXECUTOR)),
    Cmp.GT,
    ValueRef.count_selector(CardSelector.hand(EXECUTOR)),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "calendar-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "calendar-effect"),),
    (
        ConditionNode("calendar-effect", _MORE_SCORED_THAN_HELD, "draw-two-threes"),
        TimesNode("draw-two-threes", ValueRef.literal(2), "draw-three", maximum_iterations=2),
        DrawNode("draw-three", ValueRef.literal(3), "drawn-three", player=EXECUTOR),
    ),
)
