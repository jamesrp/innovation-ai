"""VACCINATION - demand return every lowest score card and, if any returned, draw
and meld a 6; afterward the activator draws and melds a 7 iff the demand returned a card.

The lowest set is snapshotted and returned completely, with the victim choosing only same-age
supply order.  The second ordinal reads root dogma change history: the demand has no possible
change unless a lowest card was returned, and its conditional draw/meld can occur only after such
a return.
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Extreme,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("vaccination")


def _demand_returned_a_card(state: Any, context: Any, registry: Any) -> bool:
    """Return whether Vaccination's demand caused any qualifying gameplay change."""

    del context, registry
    return any(
        variable.name == "dogma:qualifying-change-count"
        and isinstance(variable.value, int)
        and not isinstance(variable.value, bool)
        and variable.value > 0
        for variable in state.effect_variables
    )


PREDICATES: Final[dict[str, NamedPredicate]] = {
    "demand-returned-a-card": _demand_returned_a_card,
}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "vaccination-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "vaccination-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "vaccination-follow-up"),
    ),
    (
        SequenceNode(
            "vaccination-demand",
            (
                "snapshot-lowest",
                "order-lowest",
                "return-lowest",
                "if-returned-lowest",
            ),
        ),
        LetNode(
            "snapshot-lowest",
            "lowest-cards",
            cards=CardSelector.score(EXECUTOR, extreme=Extreme.LOWEST),
        ),
        ChoiceNode(
            "order-lowest",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("lowest-cards"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-lowest",
            MovementKind.RETURN,
            CardSelector.from_variable("lowest-cards"),
            moved_variable="returned-lowest",
            order_variable="return-order",
        ),
        ConditionNode(
            "if-returned-lowest",
            Predicate.truthy("returned-lowest"),
            "draw-and-meld-six",
        ),
        BatchNode("draw-and-meld-six", ("draw-six", "meld-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        MoveNode(
            "meld-six",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-six"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "vaccination-follow-up",
            Predicate.named("demand-returned-a-card"),
            "draw-and-meld-seven",
        ),
        BatchNode("draw-and-meld-seven", ("draw-seven", "meld-seven")),
        DrawNode("draw-seven", ValueRef.literal(7), "drawn-seven", player=EXECUTOR),
        MoveNode(
            "meld-seven",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-seven"),
            destination_player=EXECUTOR,
        ),
    ),
)
