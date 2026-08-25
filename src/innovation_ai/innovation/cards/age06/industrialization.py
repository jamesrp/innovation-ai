"""INDUSTRIALIZATION - draw and tuck one 6 for each colour currently showing a
factory, then optionally splay red or purple right.

``TimesNode`` snapshots the colour count on entry.  Factories exposed by the instruction's own
new tucks therefore cannot increase the number of iterations (rules decision 17).
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
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("industrialization")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "industrialization-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "industrialization-tucks"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "industrialization-splay"),
    ),
    (
        TimesNode(
            "industrialization-tucks",
            ValueRef(ValueRefKind.COLORS_WITH_ICON, icon=Icon.FACTORY, player=EXECUTOR),
            "draw-and-tuck-six",
        ),
        BatchNode("draw-and-tuck-six", ("draw-six", "tuck-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        MoveNode(
            "tuck-six",
            MovementKind.TUCK,
            CardSelector.from_variable("drawn-six"),
            destination_player=EXECUTOR,
        ),
        SequenceNode("industrialization-splay", ("choose-splay", "splay-right")),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.COLOR,
            "splay-color",
            colors=(Color.RED, Color.PURPLE),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-right",
            EXECUTOR,
            color_variable="splay-color",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
