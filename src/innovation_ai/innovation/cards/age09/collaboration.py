"""COLLABORATION - demand two revealed 9s, then an activator-owned transfer choice;
if an executor has ten green board cards, that player wins.

The demand snapshots the victim's hand before drawing.  The cards not in that snapshot are exactly
the two face-up draws, so the activator can choose one without learning any unrelated hand cards.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    ValueRef,
    WinNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("collaboration")

_DRAWN_CARDS: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    exclude_variable="original-hand",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "collaboration-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "collaboration-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "collaboration-win"),
    ),
    (
        SequenceNode(
            "collaboration-demand",
            (
                "snapshot-hand",
                "draw-and-reveal",
                "choose-transfer",
                "transfer-choice",
                "meld-other",
            ),
        ),
        LetNode("snapshot-hand", "original-hand", cards=CardSelector.hand(EXECUTOR)),
        BatchNode(
            "draw-and-reveal",
            ("draw-first", "draw-second", "reveal-first", "reveal-second"),
        ),
        DrawNode("draw-first", ValueRef.literal(9), "first-draw", player=EXECUTOR),
        DrawNode("draw-second", ValueRef.literal(9), "second-draw", player=EXECUTOR),
        RevealNode("reveal-first", CardSelector.from_variable("first-draw")),
        RevealNode("reveal-second", CardSelector.from_variable("second-draw")),
        ChoiceNode(
            "choose-transfer",
            ChoiceKind.CARD,
            "transferred-card",
            chooser=ACTIVATOR,
            cards=_DRAWN_CARDS,
        ),
        MoveNode(
            "transfer-choice",
            MovementKind.TRANSFER,
            CardSelector.from_variable("transferred-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
        ),
        MoveNode(
            "meld-other",
            MovementKind.MELD,
            _DRAWN_CARDS,
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "collaboration-win",
            Predicate.count(
                ValueRef.count_selector(CardSelector.stack(EXECUTOR, color=Color.GREEN)),
                Cmp.GE,
                ValueRef.literal(10),
            ),
            "win",
        ),
        WinNode("win", player=EXECUTOR),
    ),
)
