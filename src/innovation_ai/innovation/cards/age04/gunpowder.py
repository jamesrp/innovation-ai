"""GUNPOWDER - demand one castle top; a successful demand draws and scores a 2."""

from __future__ import annotations

from typing import Any, Final

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
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("gunpowder")


def _demand_transferred_a_card(state: Any, context: Any, registry: Any) -> bool:
    """Whether the preceding demand caused at least one gameplay change."""

    del context, registry
    return any(
        variable.name == "dogma:qualifying-change-count"
        and isinstance(variable.value, int)
        and not isinstance(variable.value, bool)
        and variable.value > 0
        for variable in state.effect_variables
    )


PREDICATES: Final = {"demand-transferred-a-card": _demand_transferred_a_card}

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
        ),
        ConditionNode(
            "gunpowder-follow-up",
            Predicate.named("demand-transferred-a-card"),
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
