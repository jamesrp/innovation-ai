"""SOCIETIES - "I demand you transfer a top card with a bulb higher than my top card
of the same color from your board to my board! If you do, draw a 5!"

Rules decision 8's authoritative wording is explicit: only top cards are candidates. A missing
same-colour top card on the activator's board has value zero, so any positive-valued candidate of
that colour qualifies. The victim chooses among the public qualifying tops.
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
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

CARD_ID: Final[CardId] = CardId("societies")


def _higher_than_activator_top_of_same_color(state: Any, context: Any, registry: Any) -> bool:
    """Compare the reserved selector candidate with the activator's same-colour top card."""

    raw_candidate = get_effect_variable(state, context, "_candidate")
    if not isinstance(raw_candidate, str):
        return False
    candidate = registry.card(CardId(raw_candidate))
    activator_top = state.player(context.dogma_activator).board.stack(candidate.color).top
    activator_value = 0 if activator_top is None else registry.card(activator_top).age
    return bool(candidate.age > activator_value)


PREDICATES: Final = {
    "higher-than-activator-top-of-same-color": _higher_than_activator_top_of_same_color
}

_QUALIFYING_TOPS: Final = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    icon=Icon.LIGHTBULB,
    predicate="higher-than-activator-top-of-same-color",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "societies-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "societies-demand"),),
    (
        SequenceNode(
            "societies-demand",
            ("choose-qualifying-top", "transfer-top", "if-transferred"),
        ),
        ChoiceNode(
            "choose-qualifying-top",
            ChoiceKind.CARD,
            "qualifying-top",
            chooser=EXECUTOR,
            cards=_QUALIFYING_TOPS,
        ),
        MoveNode(
            "transfer-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("qualifying-top"),
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.BOARD,
            result_variable="transferred",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("transferred"),
            "draw-five",
        ),
        DrawNode("draw-five", ValueRef.literal(5), "drawn-five", player=EXECUTOR),
    ),
)
