"""METRIC SYSTEM - if green is already splayed right, optionally splay any board
colour right; then optionally splay green right.
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    NamedPredicate,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("metric-system")


def _green_is_splayed_right(state: Any, context: Any, registry: Any) -> bool:
    """Return whether the executor's green stack currently has a right splay."""

    del registry
    return state.player(context.executor).board.stack(Color.GREEN).splay is SplayDirection.RIGHT


PREDICATES: Final[dict[str, NamedPredicate]] = {
    "green-is-splayed-right": _green_is_splayed_right,
}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "metric-system-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "metric-system-any-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "metric-system-green-splay"),
    ),
    (
        ConditionNode(
            "metric-system-any-splay",
            Predicate.named("green-is-splayed-right"),
            "choose-any-splay-sequence",
        ),
        SequenceNode(
            "choose-any-splay-sequence",
            ("choose-any-color", "splay-any-color"),
        ),
        ChoiceNode(
            "choose-any-color",
            ChoiceKind.COLOR,
            "any-color",
            color_source=ChoiceColorSource.PRESENT_ON_BOARD,
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-any-color",
            EXECUTOR,
            color_variable="any-color",
            direction=SplayDirection.RIGHT,
        ),
        SequenceNode(
            "metric-system-green-splay",
            ("choose-green", "splay-green"),
        ),
        ChoiceNode(
            "choose-green",
            ChoiceKind.COLOR,
            "green-color",
            colors=(Color.GREEN,),
            optional=True,
            minimum_stack_size=1,
        ),
        SplayNode(
            "splay-green",
            EXECUTOR,
            color_variable="green-color",
            direction=SplayDirection.RIGHT,
        ),
    ),
)
