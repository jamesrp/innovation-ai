"""ROCKETRY - return one opponent score card per two visible clocks.

Each hidden two-stage choice is collected against the unchanged score pile. The owner then orders
the complete selected subset before one grouped return.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    CollectNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("rocketry")

_UNSELECTED_OPPONENT_SCORES: Final = CardSelector(
    CardSelectorKind.SCORE,
    OPPONENT,
    exclude_variable="selected-opponent-scores",
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "rocketry-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "rocketry-effect"),),
    (
        SequenceNode(
            "rocketry-effect",
            ("collect-score-cards", "order-score-cards", "return-score-cards"),
        ),
        TimesNode(
            "collect-score-cards",
            ValueRef.icon_count(Icon.CLOCK, EXECUTOR, per=2),
            "collect-one-score",
        ),
        SequenceNode("collect-one-score", ("choose-opponent-score", "remember-score")),
        ChoiceNode(
            "choose-opponent-score",
            ChoiceKind.HIDDEN_CARD,
            "opponent-score",
            chooser=EXECUTOR,
            cards=_UNSELECTED_OPPONENT_SCORES,
            owner=OPPONENT,
        ),
        CollectNode("remember-score", "opponent-score", "selected-opponent-scores"),
        ChoiceNode(
            "order-score-cards",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=OPPONENT,
            cards=CardSelector.from_variable("selected-opponent-scores"),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-score-cards",
            MovementKind.RETURN,
            CardSelector.from_variable("selected-opponent-scores"),
            order_variable="return-order",
        ),
    ),
)
