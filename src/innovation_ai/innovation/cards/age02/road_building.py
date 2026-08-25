"""ROAD BUILDING - "Meld one or two cards from your hand. If you melded two, you may
transfer your top red card to another player's board. If you do, transfer that player's top green
card to your board."

The mandatory bounded choice permits stopping after one card even when more are available.  A
separate color-group order choice controls same-color melds.  Only two cards actually melded open
the optional red transfer, and the reciprocal green transfer executes as far as possible.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    StackPosition,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("road-building")

_MELDED_TWO: Final[Predicate] = Predicate.count(
    ValueRef.count("melded-cards"),
    Cmp.EQ,
    ValueRef.literal(2),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "road-building-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "road-building-effect"),),
    (
        SequenceNode(
            "road-building-effect",
            ("choose-melds", "order-melds", "meld-cards", "if-two-melded"),
        ),
        ChoiceNode(
            "choose-melds",
            ChoiceKind.BOUNDED_CARDS,
            "selected-melds",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=1,
            maximum=2,
        ),
        ChoiceNode(
            "order-melds",
            ChoiceKind.ORDER_CARDS,
            "meld-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-melds"),
            order_group=OrderGroup.COLOR,
        ),
        MoveNode(
            "meld-cards",
            MovementKind.MELD,
            CardSelector.from_variable("selected-melds"),
            destination_player=EXECUTOR,
            moved_variable="melded-cards",
            order_variable="meld-order",
        ),
        ConditionNode("if-two-melded", _MELDED_TWO, "if-red-top"),
        ConditionNode(
            "if-red-top",
            Predicate.non_empty(
                CardSelector.stack(EXECUTOR, color=Color.RED, position=StackPosition.TOP)
            ),
            "offer-red-transfer",
        ),
        SequenceNode(
            "offer-red-transfer",
            ("choose-red-transfer", "if-red-transfer"),
        ),
        ChoiceNode(
            "choose-red-transfer",
            ChoiceKind.BRANCH,
            "red-transfer-choice",
            branches=("transfer-top-red",),
            optional=True,
        ),
        ConditionNode(
            "if-red-transfer",
            Predicate.truthy("red-transfer-choice"),
            "transfer-red-then-green",
        ),
        SequenceNode(
            "transfer-red-then-green",
            ("transfer-top-red", "if-red-transferred"),
        ),
        MoveNode(
            "transfer-top-red",
            MovementKind.TRANSFER,
            CardSelector.stack(EXECUTOR, color=Color.RED, position=StackPosition.TOP),
            destination_player=OPPONENT,
            destination_zone=ZoneKind.BOARD,
            result_variable="red-transferred",
        ),
        ConditionNode(
            "if-red-transferred",
            Predicate.truthy("red-transferred"),
            "transfer-top-green",
        ),
        MoveNode(
            "transfer-top-green",
            MovementKind.TRANSFER,
            CardSelector.stack(OPPONENT, color=Color.GREEN, position=StackPosition.TOP),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.BOARD,
        ),
    ),
)
