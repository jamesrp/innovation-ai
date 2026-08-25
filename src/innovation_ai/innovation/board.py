"""Board geometry, splaying, and card-value queries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.state import Board, ColorStack, PlayerState
from innovation_ai.innovation.types import CardId, Color, Icon, IconSlot, SplayDirection

_COVERED_VISIBLE_SLOTS: dict[SplayDirection, tuple[IconSlot, ...]] = {
    SplayDirection.NONE: (),
    SplayDirection.LEFT: (IconSlot.BOTTOM_RIGHT,),
    SplayDirection.RIGHT: (IconSlot.TOP_LEFT, IconSlot.BOTTOM_LEFT),
    SplayDirection.UP: (
        IconSlot.BOTTOM_LEFT,
        IconSlot.BOTTOM_CENTER,
        IconSlot.BOTTOM_RIGHT,
    ),
}


def normalize_stack(stack: ColorStack) -> ColorStack:
    """Collapse a zero- or one-card stack to the mandatory unsplayed state."""

    splay = stack.splay if len(stack.cards) >= 2 else SplayDirection.NONE
    return ColorStack(stack.color, stack.cards, splay)


def splay_stack(stack: ColorStack, direction: SplayDirection) -> ColorStack:
    """Return a stack splayed in ``direction``, respecting stack collapse."""

    if len(stack.cards) <= 1:
        return ColorStack(stack.color, stack.cards, SplayDirection.NONE)
    return ColorStack(stack.color, stack.cards, direction)


def splay_board(board: Board, color: Color, direction: SplayDirection) -> Board:
    """Return a board after changing one stack's splay direction."""

    return board.replace_stack(splay_stack(board.stack(color), direction))


def top_cards(board: Board) -> tuple[CardId, ...]:
    """Return all top cards in canonical color order."""

    return tuple(stack.top for stack in board.stacks if stack.top is not None)


def top_card(board: Board, color: Color) -> CardId | None:
    """Return the top card of a color, or ``None`` when absent."""

    return board.stack(color).top


def bottom_card(board: Board, color: Color) -> CardId | None:
    """Return the bottom card of a color, or ``None`` when absent."""

    return board.stack(color).bottom


def card_beneath_top(board: Board, color: Color) -> CardId | None:
    """Return the card immediately beneath a color's top card."""

    return board.stack(color).beneath_top


def immediately_beneath(stack: ColorStack, card_id: CardId) -> CardId | None:
    """Return the card immediately beneath ``card_id`` in a stack."""

    try:
        index = stack.cards.index(card_id)
    except ValueError as error:
        raise ValueError(f"card {card_id} is not in the {stack.color} stack") from error
    return stack.cards[index - 1] if index > 0 else None


def cards_beneath(stack: ColorStack, card_id: CardId) -> tuple[CardId, ...]:
    """Return all cards beneath ``card_id`` in bottom-to-top order."""

    try:
        index = stack.cards.index(card_id)
    except ValueError as error:
        raise ValueError(f"card {card_id} is not in the {stack.color} stack") from error
    return stack.cards[:index]


def covered_visible_slots(direction: SplayDirection) -> tuple[IconSlot, ...]:
    """Return icon positions exposed on every covered card for a splay."""

    return _COVERED_VISIBLE_SLOTS[direction]


def visible_icons_for_stack(stack: ColorStack, registry: CardRegistry) -> Counter[Icon]:
    """Count visible functional icons in one color stack."""

    result: Counter[Icon] = Counter()
    if not stack.cards:
        return result

    top = registry.card(stack.cards[-1])
    result.update(top.functional_icons)
    slots = covered_visible_slots(stack.splay)
    for card_id in stack.cards[:-1]:
        card = registry.card(card_id)
        result.update(icon for slot in slots if (icon := card.icon_at(slot)) is not None)
    return result


def visible_icons(board: Board, registry: CardRegistry) -> Counter[Icon]:
    """Count all currently visible functional icons on a board."""

    result: Counter[Icon] = Counter()
    for stack in board.stacks:
        result.update(visible_icons_for_stack(stack, registry))
    return result


def score_value(player: PlayerState, registry: CardRegistry) -> int:
    """Return the sum of card values in a player's score pile."""

    return sum(registry.card(card_id).age for card_id in player.score_pile)


def card_value(card_id: CardId | None, registry: CardRegistry) -> int:
    """Return a card's value, treating an absent card as value zero."""

    return 0 if card_id is None else registry.card(card_id).age


def highest_value(cards: Iterable[CardId], registry: CardRegistry) -> int:
    """Return the highest value in ``cards``, or zero for an empty collection."""

    return max((registry.card(card_id).age for card_id in cards), default=0)


def lowest_value(cards: Iterable[CardId], registry: CardRegistry) -> int:
    """Return the lowest value in ``cards``, or zero for an empty collection."""

    return min((registry.card(card_id).age for card_id in cards), default=0)


def cards_with_highest_value(cards: Iterable[CardId], registry: CardRegistry) -> tuple[CardId, ...]:
    """Return every highest-valued card, preserving input order."""

    materialized = tuple(cards)
    value = highest_value(materialized, registry)
    return tuple(card_id for card_id in materialized if registry.card(card_id).age == value)


def cards_with_lowest_value(cards: Iterable[CardId], registry: CardRegistry) -> tuple[CardId, ...]:
    """Return every lowest-valued card, preserving input order."""

    materialized = tuple(cards)
    if not materialized:
        return ()
    value = lowest_value(materialized, registry)
    return tuple(card_id for card_id in materialized if registry.card(card_id).age == value)


def highest_top_value(board: Board, registry: CardRegistry) -> int:
    """Return the highest top-card value, or zero for an empty board."""

    return highest_value(top_cards(board), registry)


def cards_of_color(
    cards: Iterable[CardId], color: Color, registry: CardRegistry
) -> tuple[CardId, ...]:
    """Select cards of one color while preserving order."""

    return tuple(card_id for card_id in cards if registry.card(card_id).color is color)


def cards_of_value(
    cards: Iterable[CardId], value: int, registry: CardRegistry
) -> tuple[CardId, ...]:
    """Select cards of one value while preserving order."""

    return tuple(card_id for card_id in cards if registry.card(card_id).age == value)


def cards_with_icon(
    cards: Iterable[CardId], icon: Icon, registry: CardRegistry
) -> tuple[CardId, ...]:
    """Select cards whose face contains ``icon`` while preserving order."""

    return tuple(card_id for card_id in cards if icon in registry.card(card_id).functional_icons)
