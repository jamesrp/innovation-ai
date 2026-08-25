"""OPTICS — draw-and-meld a 3, then score a 4 or transfer score to a poorer opponent."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    BatchNode,
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
    ValueRefKind,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("optics")

_EXECUTOR_SCORE: Final = ValueRef(ValueRefKind.SCORE, player=EXECUTOR)
_OPPONENT_SCORE: Final = ValueRef(ValueRefKind.SCORE, player=OPPONENT)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "optics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "optics-effect"),),
    (
        SequenceNode("optics-effect", ("draw-meld-three", "crown-branch")),
        BatchNode("draw-meld-three", ("draw-three", "meld-three")),
        DrawNode("draw-three", ValueRef.literal(3), "drawn-three", player=EXECUTOR),
        MoveNode(
            "meld-three",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-three"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "crown-branch",
            Predicate.card_has_icon("drawn-three", Icon.CROWN),
            "draw-score-four",
            "if-poorer-opponent",
        ),
        BatchNode("draw-score-four", ("draw-four", "score-four")),
        DrawNode("draw-four", ValueRef.literal(4), "drawn-four", player=EXECUTOR),
        MoveNode(
            "score-four",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-four"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "if-poorer-opponent",
            Predicate.count(_EXECUTOR_SCORE, Cmp.GT, _OPPONENT_SCORE),
            "transfer-score",
        ),
        SequenceNode("transfer-score", ("choose-score-card", "give-score-card")),
        ChoiceNode(
            "choose-score-card",
            ChoiceKind.CARD,
            "score-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR),
        ),
        MoveNode(
            "give-score-card",
            MovementKind.TRANSFER,
            CardSelector.from_variable("score-card"),
            destination_player=OPPONENT,
            destination_zone=ZoneKind.SCORE,
        ),
    ),
)
