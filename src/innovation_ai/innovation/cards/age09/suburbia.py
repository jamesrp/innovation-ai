"""SUBURBIA - optionally tuck any subset of the hand, then draw and score a 1 for
each card actually selected for tucking.

Subset selection is canonical; the executor separately orders cards entering a common color
stack, and the reward count is snapshotted before any tuck changes the board.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("suburbia")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "suburbia-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "suburbia-effect"),),
    (
        SequenceNode(
            "suburbia-effect",
            ("choose-tucks", "snapshot-count", "order-tucks", "tuck-cards", "score-rewards"),
        ),
        ChoiceNode(
            "choose-tucks",
            ChoiceKind.BOUNDED_CARDS,
            "tucked-cards",
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=105,
        ),
        LetNode("snapshot-count", "tuck-count", value=ValueRef.count("tucked-cards")),
        ChoiceNode(
            "order-tucks",
            ChoiceKind.ORDER_CARDS,
            "tuck-order",
            cards=CardSelector.from_variable("tucked-cards"),
            order_group=OrderGroup.COLOR,
        ),
        MoveNode(
            "tuck-cards",
            MovementKind.TUCK,
            CardSelector.from_variable("tucked-cards"),
            destination_player=EXECUTOR,
            order_variable="tuck-order",
        ),
        TimesNode("score-rewards", ValueRef.from_variable("tuck-count"), "draw-and-score"),
        BatchNode("draw-and-score", ("draw-one", "score-one")),
        DrawNode("draw-one", ValueRef.literal(1), "reward-card", player=EXECUTOR),
        MoveNode(
            "score-one",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
