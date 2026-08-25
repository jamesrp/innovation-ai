"""OARS - repeat the crown transfer demand; draw when no transfer occurred.

The demand's movement result is accumulated in the causal scope for this exact card execution.
That scope survives direct printed-effect scheduling but is isolated for every nested execution,
where the skipped demand therefore has no transfer result.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MovementResultMode,
    MoveNode,
    Predicate,
    ProgramEffect,
    RepeatNode,
    SequenceNode,
    ValueRef,
    VariableScope,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("oars")


EFFECTS: Final[EffectProgram] = EffectProgram(
    "oars-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "oars-repeat"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "oars-fallback"),
    ),
    (
        RepeatNode(
            "oars-repeat",
            "oars-body",
            Predicate.truthy("crown-card"),
            maximum_iterations=105,
        ),
        SequenceNode("oars-body", ("choose-crown", "transfer-crown", "if-transferred")),
        ChoiceNode(
            "choose-crown",
            ChoiceKind.HIDDEN_CARD,
            "crown-card",
            chooser=EXECUTOR,
            owner=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR, icon=Icon.CROWN),
        ),
        MoveNode(
            "transfer-crown",
            MovementKind.TRANSFER,
            CardSelector.from_variable("crown-card"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="oars-demand-transferred",
            result_scope=VariableScope.CARD_EXECUTION,
            result_mode=MovementResultMode.ANY,
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("crown-card"),
            "draw-one",
        ),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ConditionNode(
            "oars-fallback",
            Predicate.negate(
                Predicate.truthy(
                    "oars-demand-transferred",
                    scope=VariableScope.CARD_EXECUTION,
                )
            ),
            "fallback-draw",
        ),
        DrawNode("fallback-draw", ValueRef.literal(1), "fallback", player=EXECUTOR),
    ),
)
