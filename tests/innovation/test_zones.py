from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    SetupProvenance,
    TerminalState,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import (
    CardLocation,
    ChangeKind,
    Placement,
    StateInvariantError,
    ZoneKind,
    ZoneOperationError,
    all_card_locations,
    assert_state_invariants,
    cards_at,
    draw_card,
    exchange_cards,
    locate_card,
    meld_card,
    move_card,
    next_draw_age,
    rearrange_stack,
    remove_card,
    return_card,
    score_card,
    set_splay,
    tuck_card,
)


def _movable_cards(
    state: GameState, registry: CardRegistry, color: Color, count: int
) -> tuple[CardId, ...]:
    cards = tuple(
        card.id
        for card in registry.cards
        if card.color is color
        and locate_card(state, card.id).kind is not ZoneKind.NORMAL_ACHIEVEMENT
    )
    assert len(cards) >= count
    return cards[:count]


def test_locations_cover_each_card_exactly_once() -> None:
    registry = load_card_registry()
    state = build_setup_state(101, registry)
    locations = all_card_locations(state)

    assert len(locations) == 105
    assert set(locations) == set(registry.by_id)
    assert sum(location.kind is ZoneKind.NORMAL_ACHIEVEMENT for location in locations.values()) == 9
    for card_id in state.players[0].hand:
        assert locate_card(state, card_id) == CardLocation.hand(PlayerId.PLAYER_1)


@pytest.mark.parametrize("requested", [-5, 0, 1])
def test_value_zero_draw_starts_at_age_one(requested: int) -> None:
    state = build_setup_state(11)
    assert next_draw_age(state, requested) == 1


def test_draw_uses_upward_only_fallback_and_return_goes_to_bottom() -> None:
    registry = load_card_registry()
    state = build_setup_state(12, registry)

    while state.supply.pile(1):
        state, change, result = draw_card(state, 1, PlayerId.PLAYER_1, registry)
        assert change.kind is ChangeKind.DRAW
        assert result.actual_age == 1
    assert next_draw_age(state, 1) == 2

    state, _, fallback = draw_card(state, 1, PlayerId.PLAYER_1, registry)
    assert fallback.actual_age == 2
    assert fallback.card_id is not None
    old_age_two = state.supply.pile(2)
    state, returned = return_card(state, fallback.card_id, registry)
    assert returned.kind is ChangeKind.RETURN
    assert state.supply.pile(2) == (*old_age_two, fallback.card_id)
    assert_state_invariants(state, registry)

    exhausted = replace(
        state,
        supply=replace(state.supply, piles=tuple(() for _ in range(10))),
        removed_cards=state.removed_cards
        + tuple(card for pile in state.supply.piles for card in pile),
    )
    assert_state_invariants(exhausted, registry)
    same, no_change, beyond = draw_card(exhausted, 10, PlayerId.PLAYER_1, registry)
    assert same is exhausted
    assert not no_change.changed
    assert beyond.beyond_age_ten


def test_meld_tuck_and_stack_collapse_preserve_geometry_correctly() -> None:
    registry = load_card_registry()
    state = build_setup_state(13, registry)
    first, second, third, fourth = _movable_cards(state, registry, Color.RED, 4)

    state, _ = meld_card(state, PlayerId.PLAYER_1, first, registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, second, registry)
    state, _ = set_splay(state, PlayerId.PLAYER_1, Color.RED, SplayDirection.RIGHT, registry)
    state, meld = meld_card(state, PlayerId.PLAYER_1, third, registry)
    assert meld.kind is ChangeKind.MELD
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.RED).cards == (first, second, third)
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.RED).splay is SplayDirection.RIGHT

    state, tuck = tuck_card(state, PlayerId.PLAYER_1, fourth, registry)
    assert tuck.kind is ChangeKind.TUCK
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.RED).cards == (
        fourth,
        first,
        second,
        third,
    )

    state, _ = score_card(state, PlayerId.PLAYER_1, fourth, registry)
    state, _ = score_card(state, PlayerId.PLAYER_1, first, registry)
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.RED).splay is SplayDirection.RIGHT
    state, _ = score_card(state, PlayerId.PLAYER_1, second, registry)
    stack = state.player(PlayerId.PLAYER_1).board.stack(Color.RED)
    assert stack.cards == (third,)
    assert stack.splay is SplayDirection.NONE

    state, _ = meld_card(state, PlayerId.PLAYER_1, fourth, registry)
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.RED).splay is SplayDirection.NONE
    assert_state_invariants(state, registry)


def test_exchange_operates_when_one_side_is_empty() -> None:
    registry = load_card_registry()
    state = build_setup_state(14, registry)
    second_hand = state.player(PlayerId.PLAYER_2).hand
    for card_id in second_hand:
        state, _ = score_card(state, PlayerId.PLAYER_2, card_id, registry)
    assert not state.player(PlayerId.PLAYER_2).hand

    first_hand = state.player(PlayerId.PLAYER_1).hand
    state, change = exchange_cards(
        state,
        CardLocation.hand(PlayerId.PLAYER_1),
        first_hand,
        CardLocation.hand(PlayerId.PLAYER_2),
        (),
        registry,
    )
    assert change.kind is ChangeKind.EXCHANGE
    assert change.changed
    assert not state.player(PlayerId.PLAYER_1).hand
    assert state.player(PlayerId.PLAYER_2).hand == first_hand
    assert_state_invariants(state, registry)


def test_splay_no_op_and_rearrangement_retains_direction() -> None:
    registry = load_card_registry()
    state = build_setup_state(15, registry)
    first, second, third = _movable_cards(state, registry, Color.BLUE, 3)
    for card_id in (first, second, third):
        state, _ = meld_card(state, PlayerId.PLAYER_1, card_id, registry)

    state, changed = set_splay(state, PlayerId.PLAYER_1, Color.BLUE, SplayDirection.UP, registry)
    assert changed.changed
    unchanged, no_op = set_splay(state, PlayerId.PLAYER_1, Color.BLUE, SplayDirection.UP, registry)
    assert unchanged is state
    assert not no_op.changed

    state, rearranged = rearrange_stack(
        state, PlayerId.PLAYER_1, Color.BLUE, (third, first, second), registry
    )
    stack = state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE)
    assert rearranged.kind is ChangeKind.REARRANGE
    assert stack.cards == (third, first, second)
    assert stack.splay is SplayDirection.UP

    with pytest.raises(ZoneOperationError, match="exactly the existing cards"):
        rearrange_stack(state, PlayerId.PLAYER_1, Color.BLUE, (first,), registry)


def test_transfer_remove_and_change_records_are_semantic() -> None:
    registry = load_card_registry()
    state = build_setup_state(16, registry)
    card_id = state.player(PlayerId.PLAYER_1).hand[0]
    original = state

    state, transfer = move_card(
        state,
        card_id,
        CardLocation.hand(PlayerId.PLAYER_2),
        registry,
        placement=Placement.TOP,
    )
    assert original.player(PlayerId.PLAYER_1).hand[0] == card_id
    assert transfer.card_moves[0].source == CardLocation.hand(PlayerId.PLAYER_1)
    assert transfer.card_moves[0].destination == CardLocation.hand(PlayerId.PLAYER_2)

    state, removed = remove_card(state, card_id, registry)
    assert removed.kind is ChangeKind.REMOVE
    assert locate_card(state, card_id) == CardLocation.removed()
    assert_state_invariants(state, registry)


def test_zone_address_and_operation_validation() -> None:
    registry = load_card_registry()
    state = build_setup_state(17, registry)
    with pytest.raises(ValueError, match="invalid player"):
        CardLocation(ZoneKind.HAND)
    with pytest.raises(ValueError, match="supply age"):
        CardLocation.supply(11)

    achievement = state.normal_achievements.cards[0]
    with pytest.raises(ZoneOperationError, match="cannot be moved"):
        move_card(state, achievement, CardLocation.hand(PlayerId.PLAYER_1), registry)

    hand_card = state.player(PlayerId.PLAYER_1).hand[0]
    wrong_color = next(color for color in Color if color is not registry.card(hand_card).color)
    with pytest.raises(ZoneOperationError, match="cannot enter"):
        move_card(
            state,
            hand_card,
            CardLocation.board(PlayerId.PLAYER_1, wrong_color),
            registry,
        )


def test_invariant_detects_wrong_supply_age() -> None:
    registry = load_card_registry()
    state = build_setup_state(18, registry)
    age_one = state.supply.pile(1)[0]
    age_two = state.supply.pile(2)[0]
    piles = list(state.supply.piles)
    piles[0] = (age_two, *piles[0][1:])
    piles[1] = (age_one, *piles[1][1:])
    corrupted = replace(state, supply=replace(state.supply, piles=tuple(piles)))

    with pytest.raises(StateInvariantError, match="wrong supply"):
        assert_state_invariants(corrupted, registry)


def test_tuck_and_score_update_only_qualifying_play_counters() -> None:
    registry = load_card_registry()
    state = replace(
        build_setup_state(20, registry),
        phase=GamePhase.PLAY,
        active_player=PlayerId.PLAYER_1,
        turn_number=1,
        paid_actions_remaining=2,
    )
    red, blue, green = (
        _movable_cards(state, registry, color, 1)[0]
        for color in (Color.RED, Color.BLUE, Color.GREEN)
    )

    state, _ = tuck_card(state, PlayerId.PLAYER_1, red, registry)
    state, _ = score_card(state, PlayerId.PLAYER_2, blue, registry)
    counters = state.turn_counters
    assert counters.for_player(PlayerId.PLAYER_1).tucked == 1
    assert counters.for_player(PlayerId.PLAYER_2).scored == 1

    state, _ = move_card(
        state,
        green,
        CardLocation.score(PlayerId.PLAYER_1),
        registry,
        kind=ChangeKind.TRANSFER,
    )
    assert state.turn_counters.for_player(PlayerId.PLAYER_1).scored == 0


def test_splay_collapse_is_included_in_the_movement_record() -> None:
    registry = load_card_registry()
    state = build_setup_state(21, registry)
    first, second = _movable_cards(state, registry, Color.YELLOW, 2)
    state, _ = meld_card(state, PlayerId.PLAYER_1, first, registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, second, registry)
    state, _ = set_splay(state, PlayerId.PLAYER_1, Color.YELLOW, SplayDirection.LEFT, registry)

    state, change = score_card(state, PlayerId.PLAYER_1, first, registry)
    assert change.splay_changes[0].before is SplayDirection.LEFT
    assert change.splay_changes[0].after is SplayDirection.NONE
    assert state.player(PlayerId.PLAYER_1).board.stack(Color.YELLOW).splay is SplayDirection.NONE


def test_removed_and_terminal_states_cannot_be_mutated() -> None:
    registry = load_card_registry()
    state = build_setup_state(22, registry)
    card_id = state.player(PlayerId.PLAYER_1).hand[0]
    state, _ = remove_card(state, card_id, registry)
    with pytest.raises(ZoneOperationError, match="cannot return to play"):
        move_card(state, card_id, CardLocation.hand(PlayerId.PLAYER_1), registry)

    terminal = replace(
        state,
        phase=GamePhase.TERMINAL,
        terminal_result=TerminalState("test", (PlayerId.PLAYER_1,)),
    )
    with pytest.raises(ZoneOperationError, match="terminal"):
        draw_card(terminal, 1, PlayerId.PLAYER_1, registry)
    with pytest.raises(ZoneOperationError, match="terminal"):
        set_splay(terminal, PlayerId.PLAYER_1, Color.RED, SplayDirection.LEFT, registry)


def test_exchange_requires_distinct_locations_and_handles_both_empty() -> None:
    registry = load_card_registry()
    state = build_setup_state(23, registry)
    first = CardLocation.board(PlayerId.PLAYER_1, Color.RED)
    second = CardLocation.board(PlayerId.PLAYER_2, Color.RED)
    unchanged, change = exchange_cards(state, first, (), second, (), registry)
    assert unchanged == state
    assert not change.changed

    with pytest.raises(ZoneOperationError, match="distinct locations"):
        exchange_cards(state, first, (), first, (), registry)


def test_invariant_rejects_catalog_fingerprint_mismatch() -> None:
    registry = load_card_registry()
    state = build_setup_state(24, registry)
    setup = SetupProvenance(
        state.setup.seed,
        "sha256:not-the-catalog",
        state.setup.shuffled_piles,
        state.setup.deal_sequence,
    )
    with pytest.raises(StateInvariantError, match="fingerprint"):
        assert_state_invariants(replace(state, setup=setup), registry)


def test_cards_at_exposes_canonical_zone_order() -> None:
    state = build_setup_state(19)
    assert cards_at(state, CardLocation.hand(PlayerId.PLAYER_1)) == state.players[0].hand
    assert cards_at(state, CardLocation.supply(10)) == state.supply.pile(10)
