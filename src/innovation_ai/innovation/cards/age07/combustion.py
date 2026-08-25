"""COMBUSTION - demand one score card per four activator crowns, then return the
victim's bottom red card.

Every mandatory private-score choice is collected against the unchanged original score pile;
the complete selected set then moves as one grouped transfer.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    CollectNode,
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

_UNSELECTED_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    EXECUTOR,
    exclude_variable="selected-score-cards",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "combustion-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "combustion-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "combustion-return"),
    ),
    (
        SequenceNode("combustion-demand", ("collect-demanded-scores", "transfer-scores")),
        TimesNode(
            "collect-demanded-scores",
            ValueRef.icon_count(Icon.CROWN, ACTIVATOR, per=4),
            "collect-one-score",
        ),
        SequenceNode("collect-one-score", ("choose-score-card", "remember-score-card")),
        ChoiceNode(
            "choose-score-card",
            ChoiceKind.HIDDEN_CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=_UNSELECTED_SCORES,
            owner=EXECUTOR,
        ),
        CollectNode("remember-score-card", "score-card", "selected-score-cards"),
        MoveNode(
            "transfer-scores",
            MovementKind.TRANSFER,
            CardSelector.from_variable("selected-score-cards"),
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
