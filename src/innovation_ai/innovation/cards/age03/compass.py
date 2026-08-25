"""COMPASS — demand a two-way exchange of qualifying top cards between boards."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("compass")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "compass-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "compass-demand"),),
    (
        SequenceNode(
            "compass-demand",
            ("choose-given-top", "give-top", "choose-taken-top", "take-top"),
        ),
        ChoiceNode(
            "choose-given-top",
            ChoiceKind.CARD,
            "given-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(
                EXECUTOR,
                icon=Icon.LEAF,
                exclude_colors=(Color.GREEN,),
            ),
        ),
        MoveNode(
            "give-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("given-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
        ),
        ChoiceNode(
            "choose-taken-top",
            ChoiceKind.CARD,
            "taken-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(ACTIVATOR, without_icon=Icon.LEAF),
        ),
        MoveNode(
            "take-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("taken-top"),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.BOARD,
        ),
    ),
)
