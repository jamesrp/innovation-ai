"""POTTERY - effect 1: "You may return up to three cards from your hand. If you returned any
cards, draw and score a card of value equal to the number of cards you returned."
effect 2: "Draw a 1."

Pottery is the slice's bounded-multi-select card. It pins three separate contracts:

* the subset is chosen incrementally and canonically (decision 16), so ``{A, B}`` has exactly one
  selection path;
* the *movement* order is a distinct decision, raised only when two returned cards share an age
  pile and the order therefore changes authoritative state (decisions 5 and 16);
* the drawn value is a quantity snapshot taken when the instruction begins (decision 17), so the
  returns performed by this instruction cannot change it.

It is also a two-effect card, so it exercises "complete effect 1 fully for both players before
starting effect 2".
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("pottery")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "pottery-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "pottery-return"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "pottery-draw"),
    ),
    (
        SequenceNode("pottery-return", ("choose-returns", "if-returned")),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "returned",
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=3,
        ),
        ConditionNode("if-returned", Predicate.truthy("returned"), "return-and-score"),
        SequenceNode(
            "return-and-score",
            ("snapshot-count", "order-returns", "return-cards", "draw-reward", "score-reward"),
        ),
        # Decision 17: the reward value is fixed before any card leaves the hand.
        LetNode("snapshot-count", "return-count", value=ValueRef.count("returned")),
        ChoiceNode(
            "order-returns",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("returned"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-cards",
            MovementKind.RETURN,
            CardSelector.from_variable("returned"),
            order_variable="return-order",
        ),
        DrawNode("draw-reward", ValueRef.from_variable("return-count"), "reward", player=EXECUTOR),
        MoveNode(
            "score-reward",
            MovementKind.SCORE,
            CardSelector.from_variable("reward"),
            destination_player=EXECUTOR,
        ),
        DrawNode("pottery-draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
    ),
)
