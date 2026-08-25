"""STATISTICS - effect 1: "I demand you transfer all the highest cards in your score
pile to your hand!"
effect 2: "You may splay your yellow cards right."

"All" removes any tie choice: every highest-valued card moves, while an empty score pile is a
legal no-op. The transfer stays within the demand victim's own zones.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("statistics")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "statistics-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "statistics-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "statistics-splay"),
    ),
    (
        MoveNode(
            "statistics-demand",
            MovementKind.TRANSFER,
            CardSelector.score(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ALL_TIED,
            ),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.HAND,
        ),
        SequenceNode("statistics-splay", ("choose-yellow", "splay-yellow")),
        ChoiceNode(
            "choose-yellow",
            ChoiceKind.COLOR,
            "yellow",
            colors=(Color.YELLOW,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-yellow",
            EXECUTOR,
            color_variable="yellow",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
