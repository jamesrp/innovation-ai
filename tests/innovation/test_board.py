from __future__ import annotations

from collections import Counter

import pytest

from innovation_ai.innovation.board import (
    bottom_card,
    card_beneath_top,
    card_value,
    cards_beneath,
    cards_of_color,
    cards_of_value,
    cards_with_highest_value,
    cards_with_icon,
    cards_with_lowest_value,
    covered_visible_slots,
    highest_top_value,
    highest_value,
    immediately_beneath,
    lowest_value,
    score_value,
    splay_board,
    splay_stack,
    top_card,
    top_cards,
    visible_icons,
    visible_icons_for_stack,
)
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.state import Board, ColorStack, PlayerState
from innovation_ai.innovation.types import CardId, Color, Icon, IconSlot, PlayerId, SplayDirection


@pytest.mark.parametrize(
    ("direction", "slots"),
    [
        (SplayDirection.NONE, ()),
        (SplayDirection.LEFT, (IconSlot.BOTTOM_RIGHT,)),
        (SplayDirection.RIGHT, (IconSlot.TOP_LEFT, IconSlot.BOTTOM_LEFT)),
        (
            SplayDirection.UP,
            (IconSlot.BOTTOM_LEFT, IconSlot.BOTTOM_CENTER, IconSlot.BOTTOM_RIGHT),
        ),
    ],
)
def test_covered_slot_splay_matrix(direction: SplayDirection, slots: tuple[IconSlot, ...]) -> None:
    assert covered_visible_slots(direction) == slots


def test_visible_icon_matrix_for_every_direction_and_card_slot() -> None:
    registry = load_card_registry()

    for covered in registry.cards:
        top = next(
            card for card in registry.cards if card.color is covered.color and card != covered
        )
        for direction in SplayDirection:
            stack = ColorStack(covered.color, (covered.id, top.id), direction)
            expected = Counter(top.functional_icons)
            expected.update(
                icon
                for slot in covered_visible_slots(direction)
                if (icon := covered.icon_at(slot)) is not None
            )
            assert visible_icons_for_stack(stack, registry) == expected


def test_top_bottom_beneath_and_board_icon_queries() -> None:
    registry = load_card_registry()
    archery = registry.card("ARCHERY")
    metalworking = registry.card("METALWORKING")
    oars = registry.card("OARS")
    board = Board.empty().replace_stack(
        ColorStack(Color.RED, (archery.id, metalworking.id, oars.id), SplayDirection.LEFT)
    )

    assert top_card(board, Color.RED) == oars.id
    assert bottom_card(board, Color.RED) == archery.id
    assert card_beneath_top(board, Color.RED) == metalworking.id
    assert immediately_beneath(board.stack(Color.RED), oars.id) == metalworking.id
    assert immediately_beneath(board.stack(Color.RED), archery.id) is None
    assert cards_beneath(board.stack(Color.RED), oars.id) == (archery.id, metalworking.id)
    assert top_card(board, Color.BLUE) is None
    assert top_cards(board) == (oars.id,)
    assert visible_icons(board, registry) == visible_icons_for_stack(
        board.stack(Color.RED), registry
    )


def test_splay_operations_replace_direction_and_small_stacks_stay_unsplayed() -> None:
    registry = load_card_registry()
    red_cards = tuple(card.id for card in registry.cards if card.color is Color.RED)[:2]
    board = Board.empty().replace_stack(ColorStack(Color.RED, red_cards))

    right = splay_board(board, Color.RED, SplayDirection.RIGHT)
    assert right.stack(Color.RED).splay is SplayDirection.RIGHT
    up = splay_board(right, Color.RED, SplayDirection.UP)
    assert up.stack(Color.RED).splay is SplayDirection.UP
    assert splay_stack(ColorStack(Color.BLUE), SplayDirection.LEFT).splay is SplayDirection.NONE
    one_card = ColorStack(Color.BLUE, (registry.card("WRITING").id,))
    assert splay_stack(one_card, SplayDirection.RIGHT).splay is SplayDirection.NONE


def test_value_score_and_stable_selector_queries() -> None:
    registry = load_card_registry()
    cards = (
        registry.card("ARCHERY").id,
        registry.card("CALENDAR").id,
        registry.card("EDUCATION").id,
        registry.card("POTTERY").id,
    )
    player = PlayerState(PlayerId.PLAYER_1, (), Board.empty(), cards)

    assert score_value(player, registry) == 7
    assert card_value(None, registry) == 0
    assert card_value(cards[1], registry) == 2
    assert highest_value((), registry) == 0
    assert lowest_value((), registry) == 0
    assert highest_value(cards, registry) == 3
    assert lowest_value(cards, registry) == 1
    assert cards_with_highest_value(cards, registry) == (CardId("education"),)
    assert cards_with_lowest_value(cards, registry) == (CardId("archery"), CardId("pottery"))
    assert cards_of_value(cards, 1, registry) == (CardId("archery"), CardId("pottery"))
    assert cards_of_color(cards, Color.RED, registry) == (CardId("archery"),)
    assert cards_with_icon(cards, Icon.LEAF, registry) == (
        CardId("calendar"),
        CardId("pottery"),
    )

    board = Board.empty().replace_stack(ColorStack(Color.PURPLE, (CardId("education"),)))
    assert highest_top_value(board, registry) == 3
    assert highest_top_value(Board.empty(), registry) == 0


def test_beneath_queries_reject_cards_outside_the_stack() -> None:
    stack = ColorStack(Color.RED, (CardId("archery"), CardId("oars")))
    with pytest.raises(ValueError, match="not in the red stack"):
        immediately_beneath(stack, CardId("pottery"))
    with pytest.raises(ValueError, match="not in the red stack"):
        cards_beneath(stack, CardId("pottery"))
