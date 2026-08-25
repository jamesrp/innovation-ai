"""THE PIRATE CODE - demand two low score cards, then conditionally score a crown top.

The transfer result is persisted in this card execution's causal scope, so nested execution with
its demand filtered out cannot inherit an unrelated outer mutation.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    EffectProgram,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    VariableScope,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("the-pirate-code")


EFFECTS: Final[EffectProgram] = EffectProgram(
    "the-pirate-code-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "pirate-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "pirate-follow-up"),
    ),
    (
        SequenceNode("pirate-demand", ("choose-two", "transfer-two")),
        ChoiceNode(
            "choose-two",
            ChoiceKind.BOUNDED_CARDS,
            "chosen-score-cards",
            chooser=EXECUTOR,
            cards=CardSelector.score(EXECUTOR, value=4, value_cmp=Cmp.LE),
            minimum=2,
            maximum=2,
        ),
        MoveNode(
            "transfer-two",
            MovementKind.TRANSFER,
            CardSelector.from_variable("chosen-score-cards"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            result_variable="pirate-demand-transferred",
            result_scope=VariableScope.CARD_EXECUTION,
            moved_variable="transferred-cards",
        ),
        ConditionNode(
            "pirate-follow-up",
            Predicate.truthy(
                "pirate-demand-transferred",
                scope=VariableScope.CARD_EXECUTION,
            ),
            "score-lowest-crown",
        ),
        SequenceNode(
            "score-lowest-crown",
            ("choose-lowest-crown", "score-chosen-crown"),
        ),
        ChoiceNode(
            "choose-lowest-crown",
            ChoiceKind.CARD,
            "lowest-crown",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(
                EXECUTOR,
                icon=Icon.CROWN,
                extreme=Extreme.LOWEST,
                extreme_scope=ExtremeScope.ONE_TIED,
            ),
        ),
        MoveNode(
            "score-chosen-crown",
            MovementKind.SCORE,
            CardSelector.from_variable("lowest-crown"),
            destination_player=EXECUTOR,
        ),
    ),
)
