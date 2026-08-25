"""MYSTICISM - "Draw and reveal a 1. If it is the same color as any card on your board,
meld it and draw a 1."

The relational color test includes covered board cards.  A nonmatching reveal is explicitly kept
so its transient public marker clears immediately; a matching card is melded before the extra
draw.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ConditionNode,
    DrawNode,
    EffectProgram,
    KeepNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RevealNode,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("mysticism")

_DRAWN_MATCHES_BOARD: Final[CardSelector] = CardSelector(
    CardSelectorKind.VARIABLE,
    variable="drawn",
    relation=SelectorRelation(
        SelectorRelationKind.SAME_COLOR_AS_ANY,
        CardSelector.board(EXECUTOR),
    ),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mysticism-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "mysticism-effect"),),
    (
        SequenceNode("mysticism-effect", ("draw-and-reveal", "color-branch")),
        BatchNode("draw-and-reveal", ("draw-one", "reveal-one")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        RevealNode("reveal-one", CardSelector.from_variable("drawn")),
        ConditionNode(
            "color-branch",
            Predicate.non_empty(_DRAWN_MATCHES_BOARD),
            "meld-and-draw",
            "keep-drawn",
        ),
        SequenceNode("meld-and-draw", ("meld-drawn", "draw-extra")),
        MoveNode(
            "meld-drawn",
            MovementKind.MELD,
            CardSelector.from_variable("drawn"),
            destination_player=EXECUTOR,
        ),
        DrawNode("draw-extra", ValueRef.literal(1), "extra", player=EXECUTOR),
        KeepNode("keep-drawn", CardSelector.from_variable("drawn")),
    ),
)
