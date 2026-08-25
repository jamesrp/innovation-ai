"""PHYSICS - "Draw three 6 and reveal them. If two or more of the drawn cards are the
same color, return the drawn cards and all cards in your hand. Otherwise, keep them."

A pure named predicate compares the three drawn colours. The return branch selects the executor's
entire live hand and asks for movement order only within same-age supply piles (decisions 5/16).
"""

from __future__ import annotations

from typing import Any, Final, cast

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    KeepNode,
    MovementKind,
    MoveNode,
    OrderGroup,
    Predicate,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId

CARD_ID: Final[CardId] = CardId("physics")


def _two_drawn_cards_share_a_color(state: Any, context: Any, registry: Any) -> bool:
    """Return whether at least two of the three scoped draw results have one colour."""

    raw_cards = tuple(
        get_effect_variable(state, context, variable)
        for variable in ("drawn-one", "drawn-two", "drawn-three")
    )
    if any(not isinstance(card_id, str) for card_id in raw_cards):
        return False
    colors = tuple(registry.card(CardId(cast(str, card_id))).color for card_id in raw_cards)
    return len(set(colors)) < len(colors)


PREDICATES: Final = {"two-drawn-cards-share-a-color": _two_drawn_cards_share_a_color}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "physics-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "physics-effect"),),
    (
        SequenceNode("physics-effect", ("draw-and-reveal", "duplicate-color-branch")),
        BatchNode(
            "draw-and-reveal",
            ("draw-one", "draw-two", "draw-three", "reveal-one", "reveal-two", "reveal-three"),
        ),
        DrawNode("draw-one", ValueRef.literal(6), "drawn-one", player=EXECUTOR),
        DrawNode("draw-two", ValueRef.literal(6), "drawn-two", player=EXECUTOR),
        DrawNode("draw-three", ValueRef.literal(6), "drawn-three", player=EXECUTOR),
        RevealNode("reveal-one", CardSelector.from_variable("drawn-one")),
        RevealNode("reveal-two", CardSelector.from_variable("drawn-two")),
        RevealNode("reveal-three", CardSelector.from_variable("drawn-three")),
        ConditionNode(
            "duplicate-color-branch",
            Predicate.named("two-drawn-cards-share-a-color"),
            "return-entire-hand",
            "keep-drawn",
        ),
        SequenceNode("return-entire-hand", ("order-returns", "return-cards")),
        ChoiceNode(
            "order-returns",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            order_group=OrderGroup.AGE,
        ),
        MoveNode(
            "return-cards",
            MovementKind.RETURN,
            CardSelector.hand(EXECUTOR),
            order_variable="return-order",
        ),
        BatchNode("keep-drawn", ("keep-one", "keep-two", "keep-three")),
        KeepNode("keep-one", CardSelector.from_variable("drawn-one")),
        KeepNode("keep-two", CardSelector.from_variable("drawn-two")),
        KeepNode("keep-three", CardSelector.from_variable("drawn-three")),
    ),
)
