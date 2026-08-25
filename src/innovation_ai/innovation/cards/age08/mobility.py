"""MOBILITY - demand the two highest eligible non-red top cards into the
activator's score pile, then draw an 8 if anything transferred.

The victim chooses tied public tops one at a time while both remain on the board. A pure selector
then combines the selected identities into one transfer instruction.
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
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("mobility")


def _candidate_is_selected(state: Any, context: Any, registry: Any) -> bool:
    """Whether the public top candidate is one of Mobility's two selections."""

    del registry
    candidate = get_effect_variable(state, context, "_candidate")
    selected = {
        get_effect_variable(state, context, "highest-one"),
        get_effect_variable(state, context, "highest-two"),
    }
    return isinstance(candidate, str) and candidate in selected


PREDICATES: Final = {"candidate-is-selected": _candidate_is_selected}

_FIRST_HIGHEST: Final = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    without_icon=Icon.FACTORY,
    exclude_colors=(Color.RED,),
    extreme=Extreme.HIGHEST,
    extreme_scope=ExtremeScope.ONE_TIED,
)

_SECOND_HIGHEST: Final = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    without_icon=Icon.FACTORY,
    exclude_colors=(Color.RED,),
    extreme=Extreme.HIGHEST,
    extreme_scope=ExtremeScope.ONE_TIED,
    exclude_variable="highest-one",
)

_SELECTED_TOPS: Final = CardSelector(
    CardSelectorKind.TOP_CARDS,
    EXECUTOR,
    predicate="candidate-is-selected",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "mobility-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "mobility-demand"),),
    (
        SequenceNode(
            "mobility-demand",
            ("choose-highest-one", "choose-highest-two", "transfer-selected", "if-transferred"),
        ),
        ChoiceNode(
            "choose-highest-one",
            ChoiceKind.CARD,
            "highest-one",
            chooser=EXECUTOR,
            cards=_FIRST_HIGHEST,
        ),
        ChoiceNode(
            "choose-highest-two",
            ChoiceKind.CARD,
            "highest-two",
            chooser=EXECUTOR,
            cards=_SECOND_HIGHEST,
        ),
        MoveNode(
            "transfer-selected",
            MovementKind.TRANSFER,
            _SELECTED_TOPS,
            destination_player=ACTIVATOR,
            destination_zone=ZoneKind.SCORE,
            moved_variable="transferred-tops",
        ),
        ConditionNode(
            "if-transferred",
            Predicate.truthy("transferred-tops"),
            "draw-eight",
        ),
        DrawNode("draw-eight", ValueRef.literal(8), "drawn-eight", player=EXECUTOR),
    ),
)
