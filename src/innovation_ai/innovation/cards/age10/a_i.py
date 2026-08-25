"""A.I. - draw and score a 10; if Robotics and Software are top cards anywhere,
the unique player with the lowest score wins.

Rules decision 9 explicitly allows the two required cards to be split across players' boards.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.effects import EffectContext, NamedPredicate, get_effect_variable
from innovation_ai.innovation.effects.program import (
    ALL_PLAYERS,
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    ValueRef,
    WinMetric,
    WinMode,
    WinNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("a-i")


def _is_required_top(state: Any, context: EffectContext, registry: CardRegistry) -> bool:
    """Whether the selector's current candidate is Robotics or Software."""

    del registry
    return get_effect_variable(state, context, "_candidate") in ("robotics", "software")


_REQUIRED_TOPS: Final = CardSelector(
    CardSelectorKind.TOP_CARDS,
    ALL_PLAYERS,
    predicate="required-top",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "a-i-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "a-i-score"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "a-i-win"),
    ),
    (
        BatchNode("a-i-score", ("draw-ten", "score-ten")),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn-ten", player=EXECUTOR),
        MoveNode(
            "score-ten",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-ten"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "a-i-win",
            Predicate.count(
                ValueRef.count_selector(_REQUIRED_TOPS),
                Cmp.EQ,
                ValueRef.literal(2),
            ),
            "lowest-score-wins",
        ),
        WinNode(
            "lowest-score-wins",
            mode=WinMode.UNIQUE_EXTREME,
            metric=WinMetric.SCORE,
            extreme=Extreme.LOWEST,
        ),
    ),
)

PREDICATES: Final[Mapping[str, NamedPredicate]] = {"required-top": _is_required_top}
