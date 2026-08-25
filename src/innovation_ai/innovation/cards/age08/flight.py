"""FLIGHT - if the executor's red cards are already splayed up, optionally splay
any one present color up; then optionally splay red up.
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
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("flight")


def _red_is_splayed_up(state: Any, context: Any, registry: Any) -> bool:
    """Whether the current executor's red stack satisfies Flight's prerequisite."""

    del registry
    return state.player(context.executor).board.stack(Color.RED).splay is SplayDirection.UP


PREDICATES: Final = {"red-is-splayed-up": _red_is_splayed_up}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "flight-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "flight-any-color"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "flight-red"),
    ),
    (
        ConditionNode(
            "flight-any-color",
            Predicate.named("red-is-splayed-up"),
            "choose-any-color-sequence",
        ),
        SequenceNode("choose-any-color-sequence", ("choose-any-color", "splay-any-color")),
        ChoiceNode(
            "choose-any-color",
            ChoiceKind.COLOR,
            "any-color",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            optional=True,
            color_source=ChoiceColorSource.PRESENT_ON_BOARD,
        ),
        SplayNode(
            "splay-any-color",
            EXECUTOR,
            color_variable="any-color",
            direction=SplayDirection.UP,
        ),
        SequenceNode("flight-red", ("choose-red", "splay-red")),
        ChoiceNode(
            "choose-red",
            ChoiceKind.COLOR,
            "red-color",
            chooser=EXECUTOR,
            target_player=EXECUTOR,
            colors=(Color.RED,),
            minimum_stack_size=1,
            optional=True,
        ),
        SplayNode(
            "splay-red",
            EXECUTOR,
            color_variable="red-color",
            direction=SplayDirection.UP,
        ),
    ),
)
