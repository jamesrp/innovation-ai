"""SKYSCRAPERS - demand a non-yellow clock top onto the activator's board; if it
moves, score the card immediately beneath it and return every other card in that pile.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    StackPosition,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("skyscrapers")

_VICTIM_PILE: Final = CardSelector.stack(EXECUTOR, color_variable="transferred-color")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "skyscrapers-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "skyscrapers-demand"),),
    (
        SequenceNode(
            "skyscrapers-demand",
            ("choose-clock-top", "bind-color", "transfer-clock-top", "if-transferred"),
        ),
        ChoiceNode(
            "choose-clock-top",
            ChoiceKind.CARD,
            "clock-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(
                EXECUTOR,
                icon=Icon.CLOCK,
                exclude_colors=(Color.YELLOW,),
            ),
        ),
        LetNode("bind-color", "transferred-color", color_of="clock-top"),
        MoveNode(
            "transfer-clock-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("clock-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
            result_variable="did-transfer",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("did-transfer"),
            "score-and-return-pile",
        ),
        SequenceNode(
            "score-and-return-pile",
            ("score-beneath", "order-other-cards", "return-other-cards"),
        ),
        MoveNode(
            "score-beneath",
            MovementKind.SCORE,
            CardSelector.stack(
                EXECUTOR,
                color_variable="transferred-color",
                position=StackPosition.TOP,
            ),
            destination_player=EXECUTOR,
        ),
        ChoiceNode(
            "order-other-cards",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=_VICTIM_PILE,
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-other-cards",
            MovementKind.RETURN,
            _VICTIM_PILE,
            order_variable="return-order",
        ),
    ),
)
