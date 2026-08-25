"""EDUCATION — optionally return a highest score card for a live highest-plus-two draw."""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("education")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "education-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "education-effect"),),
    (
        SequenceNode("education-effect", ("choose-return", "return-card", "if-returned")),
        ChoiceNode(
            "choose-return",
            ChoiceKind.CARD,
            "returned-card",
            chooser=EXECUTOR,
            cards=CardSelector.score(
                EXECUTOR,
                extreme=Extreme.HIGHEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
            optional=True,
        ),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-card"),
            result_variable="did-return",
        ),
        ConditionNode("if-returned", Predicate.truthy("did-return"), "draw-reward"),
        DrawNode(
            "draw-reward",
            ValueRef.selector_extreme(
                CardSelector.score(EXECUTOR),
                Extreme.HIGHEST,
                offset=2,
            ),
            "reward",
            player=EXECUTOR,
        ),
    ),
)
