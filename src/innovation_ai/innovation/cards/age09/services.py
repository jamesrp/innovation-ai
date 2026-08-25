"""SERVICES - transfer all highest victim score cards to the activator's hand; if
anything moved, the victim chooses a leafless activator top card to take into their hand.
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
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("services")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "services-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "services-demand"),),
    (
        SequenceNode("services-demand", ("transfer-highest", "if-transferred")),
        MoveNode(
            "transfer-highest",
            MovementKind.TRANSFER,
            CardSelector.score(EXECUTOR, extreme=Extreme.HIGHEST),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.HAND,
            moved_variable="transferred-scores",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("transferred-scores"),
            "take-top-sequence",
        ),
        SequenceNode("take-top-sequence", ("choose-top", "take-top")),
        ChoiceNode(
            "choose-top",
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
            destination_zone=ZoneKind.HAND,
        ),
    ),
)
