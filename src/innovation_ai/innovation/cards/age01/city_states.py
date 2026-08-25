"""CITY STATES - "I demand you transfer a top card with a castle from your board to my
board if you have at least four castle on your board! If you do, draw a 1!"

The demand condition reads the victim's live visible castle count.  The victim chooses the public
top card, the transfer lands on the activator's matching color stack, and only a successful
transfer permits the victim's draw.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("city-states")

_AT_LEAST_FOUR_CASTLES: Final[Predicate] = Predicate.count(
    ValueRef.icon_count(Icon.CASTLE, EXECUTOR),
    Cmp.GE,
    ValueRef.literal(4),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "city-states-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "city-states-demand"),),
    (
        ConditionNode(
            "city-states-demand",
            _AT_LEAST_FOUR_CASTLES,
            "choose-castle-top",
        ),
        SequenceNode("choose-castle-top", ("choose-top", "transfer-top", "if-transferred")),
        ChoiceNode(
            "choose-top",
            ChoiceKind.CARD,
            "castle-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(EXECUTOR, icon=Icon.CASTLE),
        ),
        MoveNode(
            "transfer-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("castle-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
            result_variable="transferred",
        ),
        ConditionNode("if-transferred", Predicate.truthy("transferred"), "draw-one"),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
    ),
)
