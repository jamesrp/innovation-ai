"""CANNING - optionally draw and tuck a 6, then score every factoryless top;
optionally splay yellow right.

The draw-and-tuck is one atomic instruction.  The top-card selector is evaluated afterward, so a
6 tucked into an empty stack can itself be among the cards scored by the conditional reward.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("canning")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "canning-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "canning-tuck-and-score"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "canning-splay"),
    ),
    (
        SequenceNode("canning-tuck-and-score", ("offer-tuck", "if-tuck")),
        ChoiceNode(
            "offer-tuck",
            ChoiceKind.BRANCH,
            "tuck-choice",
            branches=("draw-and-tuck",),
            optional=True,
        ),
        ConditionNode("if-tuck", Predicate.truthy("tuck-choice"), "tuck-and-score"),
        SequenceNode("tuck-and-score", ("draw-tuck-six", "score-factoryless-tops")),
        BatchNode("draw-tuck-six", ("draw-six", "tuck-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        MoveNode(
            "tuck-six",
            MovementKind.TUCK,
            CardSelector.from_variable("drawn-six"),
            destination_player=EXECUTOR,
        ),
        MoveNode(
            "score-factoryless-tops",
            MovementKind.SCORE,
            CardSelector.top_cards(EXECUTOR, without_icon=Icon.FACTORY),
            destination_player=EXECUTOR,
        ),
        SequenceNode("canning-splay", ("choose-yellow", "splay-yellow")),
        ChoiceNode(
            "choose-yellow",
            ChoiceKind.COLOR,
            "yellow-splay",
            colors=(Color.YELLOW,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-yellow",
            EXECUTOR,
            color_variable="yellow-splay",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
