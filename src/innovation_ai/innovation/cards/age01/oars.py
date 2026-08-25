"""OARS - effect 1: "I demand you transfer a card with a crown from your hand to my score
pile! If you do, draw a 1, and repeat this dogma effect!" effect 2: "If no cards were transferred
due to this demand, draw a 1."

The demand repeats until one execution cannot transfer a crown card.  The second printed effect
uses a named pure predicate because the frozen declarative vocabulary has no cross-effect demand
history expression: under Oars' legal schedule, a vulnerable demand has qualifying changes iff it
transferred at least one card, while an immune demand was skipped and therefore transferred none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from innovation_ai.innovation.effects.model import EffectContext, frozen_icon_counts
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
    MoveNode,
    NamedPredicate,
    Predicate,
    ProgramEffect,
    RepeatNode,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("oars")


def _no_cards_transferred_due_to_demand(
    state: Any,
    _context: EffectContext,
    _registry: Any,
) -> bool:
    """Read Oars' frozen schedule and root change count without mutating either."""

    frozen = frozen_icon_counts(state)
    if frozen is None or frozen[2] >= frozen[1]:
        # With no dogma frame, or when the opponent was immune, the demand did not transfer.
        return True
    for variable in state.effect_variables:
        if getattr(variable, "name", None) != "dogma:qualifying-change-count":
            continue
        value = getattr(variable, "value", None)
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return value == 0
    return True


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "no-cards-transferred-due-to-demand": cast(NamedPredicate, _no_cards_transferred_due_to_demand),
}

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
            Predicate.truthy("transferred"),
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
            result_variable="transferred",
        ),
        ConditionNode("if-transferred", Predicate.truthy("transferred"), "draw-one"),
        DrawNode("draw-one", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ConditionNode(
            "oars-fallback",
            Predicate.named("no-cards-transferred-due-to-demand"),
            "fallback-draw",
        ),
        DrawNode("fallback-draw", ValueRef.literal(1), "fallback", player=EXECUTOR),
    ),
)
