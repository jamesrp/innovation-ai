"""THE PIRATE CODE - effect 1: "I demand you transfer two cards of value 4 or less from
your score pile to my score pile!"
effect 2: "If any card was transferred due to the demand, score the lowest top card with a
crown from your board."

The demand victim chooses the exact private score cards, with mandatory partial execution when
fewer than two qualify. The subset is canonical but victim-owned (decisions 13/16). The second
ordinal reads whether the preceding demand entry caused a gameplay change in this dogma action.
"""

from __future__ import annotations

from typing import Any, Final

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
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("the-pirate-code")


def _demand_transferred_a_card(state: Any, context: Any, registry: Any) -> bool:
    """Read the root dogma change count before the first non-demand ordinal begins."""

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
            moved_variable="transferred-cards",
        ),
        ConditionNode(
            "pirate-follow-up",
            Predicate.named("demand-transferred-a-card"),
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
