"""SKYSCRAPERS - demand a non-yellow clock top onto the activator's board; if it
moves, score the card immediately beneath it and return every other card in that pile.

The beneath card and return set are snapshotted, the owner orders returns, and the score plus all
returns commit as one compound atom.
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
_VICTIM_OTHER_CARDS: Final = CardSelector(
    CardSelectorKind.BOARD_STACK,
    EXECUTOR,
    color_variable="transferred-color",
    exclude_variable="beneath-card",
)

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
            (
                "bind-beneath-card",
                "bind-other-cards",
                "order-other-cards",
                "commit-score-and-returns",
            ),
        ),
        LetNode(
            "bind-beneath-card",
            "beneath-card",
            cards=CardSelector.stack(
                EXECUTOR,
                color_variable="transferred-color",
                position=StackPosition.TOP,
            ),
        ),
        LetNode(
            "bind-other-cards",
            "other-cards",
            cards=_VICTIM_OTHER_CARDS,
        ),
        ChoiceNode(
            "order-other-cards",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("other-cards"),
            order_group=OrderGroup.AGE,
        ),
        BatchNode(
            "commit-score-and-returns",
            ("score-beneath", "return-other-cards"),
        ),
        MoveNode(
            "score-beneath",
            MovementKind.SCORE,
            CardSelector.from_variable("beneath-card"),
            destination_player=EXECUTOR,
        ),
        MoveNode(
            "return-other-cards",
            MovementKind.RETURN,
            CardSelector.from_variable("other-cards"),
            order_variable="return-order",
        ),
    ),
)
