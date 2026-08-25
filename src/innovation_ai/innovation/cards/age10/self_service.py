"""SELF SERVICE - execute another top card's non-demand effects without sharing;
then win if the executor has more achievements than every other player.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    EffectProgram,
    NestedNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ValueRefKind,
    WinNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("self-service")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "self-service-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "self-service-nested"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "self-service-win"),
    ),
    (
        SequenceNode("self-service-nested", ("choose-other-top", "execute-other-top")),
        ChoiceNode(
            "choose-other-top",
            ChoiceKind.CARD,
            "selected-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(EXECUTOR, exclude_source_card=True),
        ),
        NestedNode("execute-other-top", "selected-top"),
        ConditionNode(
            "self-service-win",
            Predicate.count(
                ValueRef(ValueRefKind.ACHIEVEMENT_COUNT, player=EXECUTOR),
                Cmp.GT,
                ValueRef(ValueRefKind.ACHIEVEMENT_COUNT, player=OPPONENT),
            ),
            "executor-wins",
        ),
        WinNode("executor-wins", player=EXECUTOR),
    ),
)
