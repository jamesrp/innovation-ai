"""CLOTHING - effect 1: "Meld a card from your hand of different color from any card on
your board." effect 2: "Draw and score a 1 for each color present on your board not present on
any other player's board."

The first selector uses the frozen relational vocabulary and is vacuously unrestricted by an
empty board.  Effect two snapshots its unique-color count on entry, then performs each
``draw and score`` as one atomic instruction.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("clothing")

_MELDABLE: Final[CardSelector] = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    relation=SelectorRelation(
        SelectorRelationKind.DIFFERENT_COLOR_FROM_ALL,
        CardSelector.board(EXECUTOR),
    ),
)

_UNIQUE_COLOR_COUNT: Final[ValueRef] = ValueRef(
    ValueRefKind.COLORS_PRESENT_ONLY_HERE,
    player=EXECUTOR,
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "clothing-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "clothing-meld"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "clothing-score"),
    ),
    (
        SequenceNode("clothing-meld", ("choose-meld", "meld-chosen")),
        ChoiceNode(
            "choose-meld",
            ChoiceKind.CARD,
            "meld-card",
            cards=_MELDABLE,
        ),
        MoveNode(
            "meld-chosen",
            MovementKind.MELD,
            CardSelector.from_variable("meld-card"),
            destination_player=EXECUTOR,
        ),
        TimesNode("clothing-score", _UNIQUE_COLOR_COUNT, "draw-score-one", maximum_iterations=5),
        BatchNode("draw-score-one", ("draw-one", "score-one")),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        MoveNode(
            "score-one",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn"),
            destination_player=EXECUTOR,
        ),
    ),
)
