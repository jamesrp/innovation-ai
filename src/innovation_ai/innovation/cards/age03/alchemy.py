"""ALCHEMY — draw and reveal a 4 per three castles; a red draw returns the whole hand.

The repeated draw/reveal is atomic per card.  A pure named selector records exactly the cards
that remain physically revealed, so the non-red branch keeps only those draws while the red
branch returns the complete hand as one movement atom.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.effects.model import EffectContext
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    KeepNode,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    OrderGroup,
    Predicate,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("alchemy")


def _candidate_is_revealed(state: Any, context: EffectContext, registry: CardRegistry) -> bool:
    """Whether the selector's reserved candidate is physically revealed right now."""

    del registry
    candidate_key = f"{context.scope}:_candidate"
    candidate = next(
        (variable.value for variable in state.effect_variables if variable.name == candidate_key),
        None,
    )
    return isinstance(candidate, str) and any(
        marker.card_id.value == candidate and marker.scope == context.scope
        for marker in state.revealed
    )


PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "candidate-is-revealed": _candidate_is_revealed,
}

_REVEALED_HAND: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    predicate="candidate-is-revealed",
)
_RED_DRAWS: Final = CardSelector(
    CardSelectorKind.VARIABLE,
    variable="drawn-cards",
    colors=(Color.RED,),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "alchemy-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "alchemy-reveal"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "alchemy-meld-score"),
    ),
    (
        SequenceNode(
            "alchemy-reveal",
            ("draw-fours", "remember-draws", "red-draw-branch"),
        ),
        TimesNode(
            "draw-fours",
            ValueRef.icon_count(Icon.CASTLE, EXECUTOR, per=3),
            "draw-and-reveal-four",
        ),
        BatchNode("draw-and-reveal-four", ("draw-four", "reveal-four")),
        DrawNode("draw-four", ValueRef.literal(4), "drawn", player=EXECUTOR),
        RevealNode("reveal-four", CardSelector.from_variable("drawn")),
        LetNode("remember-draws", "drawn-cards", cards=_REVEALED_HAND),
        ConditionNode(
            "red-draw-branch",
            Predicate.non_empty(_RED_DRAWS),
            "return-whole-hand",
            "keep-draws",
        ),
        SequenceNode(
            "return-whole-hand",
            ("order-returned-hand", "return-hand"),
        ),
        ChoiceNode(
            "order-returned-hand",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-hand",
            MovementKind.RETURN,
            CardSelector.hand(EXECUTOR),
            order_variable="return-order",
        ),
        KeepNode("keep-draws", CardSelector.from_variable("drawn-cards")),
        SequenceNode(
            "alchemy-meld-score",
            ("choose-meld", "meld-card", "choose-score", "score-card"),
        ),
        ChoiceNode(
            "choose-meld",
            ChoiceKind.CARD,
            "meld-card",
            cards=CardSelector.hand(EXECUTOR),
        ),
        MoveNode(
            "meld-card",
            MovementKind.MELD,
            CardSelector.from_variable("meld-card"),
            destination_player=EXECUTOR,
        ),
        ChoiceNode(
            "choose-score",
            ChoiceKind.CARD,
            "score-card",
            cards=CardSelector.hand(EXECUTOR),
        ),
        MoveNode(
            "score-card",
            MovementKind.SCORE,
            CardSelector.from_variable("score-card"),
            destination_player=EXECUTOR,
        ),
    ),
)
