"""MAPMAKING - demand: "I demand you transfer a 1 from your score pile, if it has any,
to my score pile!" Then: "If any card was transferred due to the demand, draw and score a 1."

The victim owns any hidden tied identity choice.  The second effect reads the dogma action's
persisted demand-change count; Mapmaking's demand has no other mutation, so a positive count is
exactly evidence that its transfer occurred.  The reward is one atomic draw-and-score.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from innovation_ai.innovation.effects.model import EffectContext, frozen_icon_counts
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
    NamedPredicate,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("mapmaking")


def _any_card_transferred_due_to_demand(
    state: Any,
    _context: EffectContext,
    _registry: Any,
) -> bool:
    """Recognize a vulnerable demand followed by at least one player-facing change."""

    if _context.nested:
        # Nested execution runs non-demand effects only, so Mapmaking's prerequisite demand did
        # not happen even when the outer dogma action already recorded unrelated changes.
        return False
    frozen = frozen_icon_counts(state)
    if frozen is None or frozen[2] >= frozen[1]:
        return False
    for variable in state.effect_variables:
        if getattr(variable, "name", None) != "dogma:qualifying-change-count":
            continue
        value = getattr(variable, "value", None)
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    return False


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "any-card-transferred-due-to-demand": cast(
        NamedPredicate,
        _any_card_transferred_due_to_demand,
    ),
}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mapmaking-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "mapmaking-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "mapmaking-reward"),
    ),
    (
        SequenceNode("mapmaking-demand", ("choose-one", "transfer-one")),
        ChoiceNode(
            "choose-one",
            ChoiceKind.HIDDEN_CARD,
            "score-one",
            chooser=EXECUTOR,
            owner=EXECUTOR,
            cards=CardSelector.score(EXECUTOR, value=1),
        ),
        MoveNode(
            "transfer-one",
            MovementKind.TRANSFER,
            CardSelector.from_variable("score-one"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
        ),
        ConditionNode(
            "mapmaking-reward",
            Predicate.named("any-card-transferred-due-to-demand"),
            "draw-score-one",
        ),
        BatchNode("draw-score-one", ("draw-one", "score-reward")),
        DrawNode("draw-one", ValueRef.literal(1), "reward-card", player=EXECUTOR),
        MoveNode(
            "score-reward",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
