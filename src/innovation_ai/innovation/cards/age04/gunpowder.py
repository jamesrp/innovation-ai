"""GUNPOWDER - demand one castle top; a successful demand draws and scores a 2."""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.model import ScopedVariables
from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
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
    ValueRef,
    VariableScope,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("gunpowder")


def _own_demand_transferred(state: Any, context: Any, registry: Any) -> bool:
    """Read only this direct Gunpowder activation's persisted demand result."""

    del registry
    if context.nested:
        return False
    root_scope = context.scope.split("/", maxsplit=1)[0]
    return (
        ScopedVariables(state.effect_variables).get(
            root_scope,
            "gunpowder-demand-transferred",
            False,
        )
        is True
    )


PREDICATES: Final = {"own-demand-transferred": _own_demand_transferred}


EFFECTS: Final[EffectProgram] = EffectProgram(
    "gunpowder-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "gunpowder-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "gunpowder-follow-up"),
    ),
    (
        SequenceNode("gunpowder-demand", ("choose-castle-top", "transfer-castle-top")),
        ChoiceNode(
            "choose-castle-top",
            ChoiceKind.CARD,
            "castle-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(EXECUTOR, icon=Icon.CASTLE),
        ),
        MoveNode(
            "transfer-castle-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("castle-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="gunpowder-demand-transferred",
            result_scope=VariableScope.ROOT,
        ),
        ConditionNode(
            "gunpowder-follow-up",
            Predicate.named("own-demand-transferred"),
            "draw-score-two",
        ),
        BatchNode("draw-score-two", ("draw-two", "score-two")),
        DrawNode("draw-two", ValueRef.literal(2), "drawn-two", player=EXECUTOR),
        MoveNode(
            "score-two",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-two"),
            destination_player=EXECUTOR,
        ),
    ),
)
