"""COMBUSTION - demand: "I demand you transfer one card from your score pile to my
score pile for every four crowns on my board!" Then: "Return your bottom red card."

The crown quantity belongs to the demand instruction and is snapshotted by ``TimesNode`` before
its first transfer. Each demanded private-score choice belongs to the victim, the zone owner.
"""

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
    StackPosition,
    TimesNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("combustion")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "combustion-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "combustion-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "combustion-return"),
    ),
    (
        TimesNode(
            "combustion-demand",
            ValueRef.icon_count(Icon.CROWN, ACTIVATOR, per=4),
            "transfer-one-sequence",
        ),
        SequenceNode("transfer-one-sequence", ("choose-score-card", "transfer-score-card")),
        ChoiceNode(
            "choose-score-card",
            ChoiceKind.HIDDEN_CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
            owner=EXECUTOR,
        ),
        MoveNode(
            "transfer-score-card",
            MovementKind.TRANSFER,
            CardSelector.from_variable("score-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
        ),
        MoveNode(
            "combustion-return",
            MovementKind.RETURN,
            CardSelector.stack(
                EXECUTOR,
                color=Color.RED,
                position=StackPosition.BOTTOM,
            ),
        ),
    ),
)
