"""INVENTION - turn one left splay right for a scored 4, then claim Wonder if eligible."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    Predicate,
    ProgramEffect,
    SequenceNode,
    SplayNode,
    ValueRef,
)
from innovation_ai.innovation.types import (
    CardId,
    DogmaEffectId,
    SpecialAchievementId,
    SplayDirection,
)

CARD_ID: Final[CardId] = CardId("invention")


def _candidate_is_splayed_left(state: Any, context: Any, registry: Any) -> bool:
    """Whether the reserved top-card candidate represents a currently left-splayed stack."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    if not isinstance(raw_candidate, str):
        return False
    color = registry.card(CardId(raw_candidate)).color
    return state.player(context.executor).board.stack(color).splay is SplayDirection.LEFT


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-splayed-left": _candidate_is_splayed_left,
}

_LEFT_SPLAYED_TOPS: Final[CardSelector] = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    predicate="candidate-is-splayed-left",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "invention-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "invention-splay"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "claim-wonder"),
    ),
    (
        SequenceNode("invention-splay", ("choose-left-stack", "if-left-stack")),
        ChoiceNode(
            "choose-left-stack",
            ChoiceKind.CARD,
            "left-stack-top",
            chooser=EXECUTOR,
            cards=_LEFT_SPLAYED_TOPS,
            optional=True,
        ),
        ConditionNode(
            "if-left-stack",
            Predicate.truthy("left-stack-top"),
            "splay-and-score",
        ),
        SequenceNode(
            "splay-and-score",
            ("bind-left-color", "splay-right", "draw-score-four"),
        ),
        LetNode("bind-left-color", "left-color", color_of="left-stack-top"),
        SplayNode(
            "splay-right",
            EXECUTOR,
            color_variable="left-color",
            direction=SplayDirection.RIGHT,
        ),
        BatchNode("draw-score-four", ("draw-four", "score-four")),
        DrawNode("draw-four", ValueRef.literal(4), "drawn-four", player=EXECUTOR),
        MoveNode(
            "score-four",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-four"),
            destination_player=EXECUTOR,
        ),
        ClaimAchievementNode(
            "claim-wonder",
            SpecialAchievementId.WONDER,
            player=EXECUTOR,
        ),
    ),
)
