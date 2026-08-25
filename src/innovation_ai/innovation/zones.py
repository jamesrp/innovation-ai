"""Pure authoritative zone operations and card-conservation checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum

from innovation_ai.innovation.board import splay_stack
from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.state import ColorStack, GamePhase, GameState
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SplayDirection,
)


class StateInvariantError(RuntimeError):
    """The authoritative state violates a core engine invariant."""


class ZoneOperationError(ValueError):
    """A requested primitive cannot legally address the specified zones."""


class ZoneKind(StrEnum):
    """Authoritative locations that can contain innovation cards."""

    SUPPLY = "supply"
    HAND = "hand"
    BOARD = "board"
    SCORE = "score"
    NORMAL_ACHIEVEMENT = "normal-achievement"
    REMOVED = "removed"


class Placement(StrEnum):
    """Position at which a card enters an ordered zone."""

    TOP = "top"
    BOTTOM = "bottom"


class ChangeKind(StrEnum):
    """Semantic kinds emitted by WP2 zone and geometry primitives."""

    DRAW = "draw"
    MELD = "meld"
    TUCK = "tuck"
    SCORE = "score"
    RETURN = "return"
    TRANSFER = "transfer"
    EXCHANGE = "exchange"
    REMOVE = "remove"
    SPLAY = "splay"
    REARRANGE = "rearrange"


@dataclass(frozen=True, slots=True)
class CardLocation:
    """Stable semantic address of a card zone."""

    kind: ZoneKind
    player_id: PlayerId | None = None
    color: Color | None = None
    age: int | None = None
    normal_achievement_id: NormalAchievementId | None = None

    def __post_init__(self) -> None:
        expected_player = self.kind in {ZoneKind.HAND, ZoneKind.BOARD, ZoneKind.SCORE}
        if (self.player_id is not None) != expected_player:
            raise ValueError(f"{self.kind} has invalid player address")
        if (self.color is not None) != (self.kind is ZoneKind.BOARD):
            raise ValueError(f"{self.kind} has invalid color address")
        if (self.age is not None) != (self.kind is ZoneKind.SUPPLY):
            raise ValueError(f"{self.kind} has invalid age address")
        if self.age is not None and not 1 <= self.age <= 10:
            raise ValueError("supply age must be 1-10")
        if (self.normal_achievement_id is not None) != (self.kind is ZoneKind.NORMAL_ACHIEVEMENT):
            raise ValueError(f"{self.kind} has invalid normal-achievement address")

    @classmethod
    def supply(cls, age: int) -> CardLocation:
        return cls(ZoneKind.SUPPLY, age=age)

    @classmethod
    def hand(cls, player_id: PlayerId) -> CardLocation:
        return cls(ZoneKind.HAND, player_id=player_id)

    @classmethod
    def board(cls, player_id: PlayerId, color: Color) -> CardLocation:
        return cls(ZoneKind.BOARD, player_id=player_id, color=color)

    @classmethod
    def score(cls, player_id: PlayerId) -> CardLocation:
        return cls(ZoneKind.SCORE, player_id=player_id)

    @classmethod
    def normal_achievement(cls, achievement_id: NormalAchievementId) -> CardLocation:
        return cls(ZoneKind.NORMAL_ACHIEVEMENT, normal_achievement_id=achievement_id)

    @classmethod
    def removed(cls) -> CardLocation:
        return cls(ZoneKind.REMOVED)


@dataclass(frozen=True, slots=True)
class CardMove:
    """One card's source and destination within an atomic change."""

    card_id: CardId
    source: CardLocation
    destination: CardLocation


@dataclass(frozen=True, slots=True)
class SplayChange:
    """A stack geometry change contained in an atomic change."""

    player_id: PlayerId
    color: Color
    before: SplayDirection
    after: SplayDirection


@dataclass(frozen=True, slots=True)
class ChangeRecord:
    """Deterministically ordered result of one semantic state primitive."""

    kind: ChangeKind
    card_moves: tuple[CardMove, ...] = ()
    splay_changes: tuple[SplayChange, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether the primitive changed cards or effective geometry."""

        return bool(self.card_moves or self.splay_changes)


@dataclass(frozen=True, slots=True)
class DrawResult:
    """Outcome of resolving the upward-only supply fallback."""

    requested_age: int
    actual_age: int | None
    card_id: CardId | None

    @property
    def beyond_age_ten(self) -> bool:
        """Whether the draw would require a nonexistent age above ten."""

        return self.actual_age is None


def cards_at(state: GameState, location: CardLocation) -> tuple[CardId, ...]:
    """Return cards at ``location`` in authoritative order."""

    if location.kind is ZoneKind.SUPPLY:
        assert location.age is not None
        return state.supply.pile(location.age)
    if location.kind is ZoneKind.REMOVED:
        return state.removed_cards
    if location.kind is ZoneKind.NORMAL_ACHIEVEMENT:
        assert location.normal_achievement_id is not None
        return (state.normal_achievements.card(location.normal_achievement_id),)

    assert location.player_id is not None
    player = state.player(location.player_id)
    if location.kind is ZoneKind.HAND:
        return player.hand
    if location.kind is ZoneKind.SCORE:
        return player.score_pile
    if location.kind is ZoneKind.BOARD:
        assert location.color is not None
        return player.board.stack(location.color).cards
    raise AssertionError(f"unhandled zone kind: {location.kind}")


def all_card_locations(state: GameState) -> dict[CardId, CardLocation]:
    """Map every uniquely located card to its semantic location."""

    result: dict[CardId, CardLocation] = {}
    for age in range(1, 11):
        location = CardLocation.supply(age)
        for card_id in cards_at(state, location):
            if card_id in result:
                raise StateInvariantError(f"card appears in multiple locations: {card_id}")
            result[card_id] = location
    for achievement_id in NormalAchievementId:
        location = CardLocation.normal_achievement(achievement_id)
        card_id = cards_at(state, location)[0]
        if card_id in result:
            raise StateInvariantError(f"card appears in multiple locations: {card_id}")
        result[card_id] = location
    for player_id in PlayerId:
        for location in (
            CardLocation.hand(player_id),
            *(CardLocation.board(player_id, color) for color in Color),
            CardLocation.score(player_id),
        ):
            for card_id in cards_at(state, location):
                if card_id in result:
                    raise StateInvariantError(f"card appears in multiple locations: {card_id}")
                result[card_id] = location
    removed = CardLocation.removed()
    for card_id in cards_at(state, removed):
        if card_id in result:
            raise StateInvariantError(f"card appears in multiple locations: {card_id}")
        result[card_id] = removed
    return result


def locate_card(state: GameState, card_id: CardId) -> CardLocation:
    """Return the unique authoritative location of a card."""

    try:
        return all_card_locations(state)[card_id]
    except KeyError as error:
        raise ZoneOperationError(f"card is not present in the game state: {card_id}") from error


def assert_state_invariants(state: GameState, registry: CardRegistry) -> None:
    """Validate conservation, unique locations, geometry, and achievement ownership."""

    locations = all_card_locations(state)
    if state.setup.card_data_fingerprint != registry.data_fingerprint:
        raise StateInvariantError("state card-data fingerprint does not match the registry")
    expected = set(registry.by_id)
    actual = set(locations)
    if actual != expected:
        missing = sorted(str(card_id) for card_id in expected - actual)
        extra = sorted(str(card_id) for card_id in actual - expected)
        raise StateInvariantError(f"card conservation failed; missing={missing}, extra={extra}")

    for age in range(1, 11):
        for card_id in state.supply.pile(age):
            if registry.card(card_id).age != age:
                raise StateInvariantError(f"card {card_id} is in the wrong supply")
    for achievement_id in NormalAchievementId:
        card_id = state.normal_achievements.card(achievement_id)
        expected_age = tuple(NormalAchievementId).index(achievement_id) + 1
        if registry.card(card_id).age != expected_age:
            raise StateInvariantError(f"card {card_id} is the wrong normal achievement")
    for player in state.players:
        for stack in player.board.stacks:
            if len(stack.cards) <= 1 and stack.splay is not SplayDirection.NONE:
                raise StateInvariantError(f"collapsed {stack.color} stack remains splayed")
            for card_id in stack.cards:
                if registry.card(card_id).color is not stack.color:
                    raise StateInvariantError(f"card {card_id} is in the wrong color stack")

    normal_claims = [item for player in state.players for item in player.normal_achievements]
    special_claims = [item for player in state.players for item in player.special_achievements]
    if len(normal_claims) != len(set(normal_claims)):
        raise StateInvariantError("a normal achievement has multiple owners")
    if len(special_claims) != len(set(special_claims)):
        raise StateInvariantError("a special achievement has multiple owners")
    if Counter(player.player_id for player in state.players) != Counter(PlayerId):
        raise StateInvariantError("player identities are invalid")


def _assert_mutable(state: GameState) -> None:
    if state.phase is GamePhase.TERMINAL:
        raise ZoneOperationError("terminal game state cannot be mutated")


def _splay_at(state: GameState, location: CardLocation) -> SplayDirection | None:
    if location.kind is not ZoneKind.BOARD:
        return None
    assert location.player_id is not None and location.color is not None
    return state.player(location.player_id).board.stack(location.color).splay


def _movement_splay_changes(
    before: GameState, after: GameState, locations: tuple[CardLocation, ...]
) -> tuple[SplayChange, ...]:
    result: list[SplayChange] = []
    seen: set[CardLocation] = set()
    for location in locations:
        if location in seen or location.kind is not ZoneKind.BOARD:
            continue
        seen.add(location)
        before_splay = _splay_at(before, location)
        after_splay = _splay_at(after, location)
        if before_splay is not after_splay:
            assert location.player_id is not None and location.color is not None
            assert before_splay is not None and after_splay is not None
            result.append(
                SplayChange(location.player_id, location.color, before_splay, after_splay)
            )
    return tuple(result)


def next_draw_age(state: GameState, requested_age: int) -> int | None:
    """Return the next non-empty supply age, searching upward only."""

    start_age = max(1, requested_age)
    if start_age > 10:
        return None
    return next((age for age in range(start_age, 11) if state.supply.pile(age)), None)


def _replace_zone_cards(
    state: GameState,
    location: CardLocation,
    cards: tuple[CardId, ...],
    registry: CardRegistry,
    *,
    preserve_splay: bool = True,
) -> GameState:
    if location.kind is ZoneKind.SUPPLY:
        assert location.age is not None
        for card_id in cards:
            if registry.card(card_id).age != location.age:
                raise ZoneOperationError(f"card {card_id} cannot enter age {location.age} supply")
        return replace(state, supply=state.supply.replace_pile(location.age, cards))
    if location.kind is ZoneKind.REMOVED:
        return replace(state, removed_cards=cards)
    if location.kind is ZoneKind.NORMAL_ACHIEVEMENT:
        raise ZoneOperationError("normal-achievement identities cannot be moved by zone primitives")

    assert location.player_id is not None
    player = state.player(location.player_id)
    if location.kind is ZoneKind.HAND:
        replacement = replace(player, hand=cards)
    elif location.kind is ZoneKind.SCORE:
        replacement = replace(player, score_pile=cards)
    elif location.kind is ZoneKind.BOARD:
        assert location.color is not None
        for card_id in cards:
            if registry.card(card_id).color is not location.color:
                raise ZoneOperationError(f"card {card_id} cannot enter {location.color} stack")
        old_stack = player.board.stack(location.color)
        direction = old_stack.splay if preserve_splay and len(cards) >= 2 else SplayDirection.NONE
        stack = ColorStack(location.color, cards, direction)
        replacement = replace(player, board=player.board.replace_stack(stack))
    else:  # pragma: no cover - exhaustive guard
        raise AssertionError(f"unhandled zone kind: {location.kind}")
    return state.replace_player(replacement)


def _remove_from_zone(
    state: GameState, location: CardLocation, card_id: CardId, registry: CardRegistry
) -> GameState:
    cards = cards_at(state, location)
    try:
        index = cards.index(card_id)
    except ValueError as error:
        raise ZoneOperationError(f"card {card_id} is not in {location}") from error
    return _replace_zone_cards(state, location, cards[:index] + cards[index + 1 :], registry)


def _add_to_zone(
    state: GameState,
    location: CardLocation,
    card_id: CardId,
    placement: Placement,
    registry: CardRegistry,
) -> GameState:
    cards = cards_at(state, location)
    if location.kind is ZoneKind.SUPPLY:
        updated = (card_id, *cards) if placement is Placement.TOP else (*cards, card_id)
    elif location.kind is ZoneKind.BOARD:
        updated = (*cards, card_id) if placement is Placement.TOP else (card_id, *cards)
    else:
        # Hands, score piles, and removed cards have no rule-relevant top/bottom access.
        updated = (*cards, card_id)
    return _replace_zone_cards(state, location, tuple(updated), registry)


def move_card(
    state: GameState,
    card_id: CardId,
    destination: CardLocation,
    registry: CardRegistry,
    *,
    kind: ChangeKind = ChangeKind.TRANSFER,
    placement: Placement = Placement.TOP,
) -> tuple[GameState, ChangeRecord]:
    """Move one card atomically and return a semantic change record."""

    _assert_mutable(state)
    source = locate_card(state, card_id)
    if source.kind is ZoneKind.NORMAL_ACHIEVEMENT:
        raise ZoneOperationError("normal-achievement identity cannot be moved")
    if source.kind is ZoneKind.REMOVED:
        raise ZoneOperationError("removed cards cannot return to play")
    if destination.kind is ZoneKind.REMOVED and kind is not ChangeKind.REMOVE:
        raise ZoneOperationError("only a remove operation can move a card out of the game")
    if kind is ChangeKind.REMOVE and destination.kind is not ZoneKind.REMOVED:
        raise ZoneOperationError("remove operations must use the removed-card destination")
    if source == destination:
        assert_state_invariants(state, registry)
        return state, ChangeRecord(kind)
    updated = _remove_from_zone(state, source, card_id, registry)
    updated = _add_to_zone(updated, destination, card_id, placement, registry)
    assert_state_invariants(updated, registry)
    splay_changes = _movement_splay_changes(state, updated, (source, destination))
    return updated, ChangeRecord(
        kind,
        (CardMove(card_id, source, destination),),
        splay_changes,
    )


def draw_card(
    state: GameState, requested_age: int, player_id: PlayerId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord, DrawResult]:
    """Draw to a hand using value-zero normalization and upward-only fallback."""

    _assert_mutable(state)
    actual_age = next_draw_age(state, requested_age)
    if actual_age is None:
        assert_state_invariants(state, registry)
        return state, ChangeRecord(ChangeKind.DRAW), DrawResult(requested_age, None, None)
    card_id = state.supply.pile(actual_age)[0]
    updated, change = move_card(
        state,
        card_id,
        CardLocation.hand(player_id),
        registry,
        kind=ChangeKind.DRAW,
        placement=Placement.BOTTOM,
    )
    return updated, change, DrawResult(requested_age, actual_age, card_id)


def meld_card(
    state: GameState, player_id: PlayerId, card_id: CardId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    """Place a card atop its matching board stack, preserving its splay."""

    color = registry.card(card_id).color
    return move_card(
        state,
        card_id,
        CardLocation.board(player_id, color),
        registry,
        kind=ChangeKind.MELD,
        placement=Placement.TOP,
    )


def tuck_card(
    state: GameState, player_id: PlayerId, card_id: CardId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    """Place a card beneath its matching board stack, preserving its splay."""

    color = registry.card(card_id).color
    updated, change = move_card(
        state,
        card_id,
        CardLocation.board(player_id, color),
        registry,
        kind=ChangeKind.TUCK,
        placement=Placement.BOTTOM,
    )
    if updated.phase is GamePhase.PLAY and change.changed:
        updated = replace(
            updated,
            turn_counters=updated.turn_counters.increment(player_id, tucked=1),
        )
    return updated, change


def score_card(
    state: GameState, player_id: PlayerId, card_id: CardId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    """Move a card into a player's score pile."""

    updated, change = move_card(
        state,
        card_id,
        CardLocation.score(player_id),
        registry,
        kind=ChangeKind.SCORE,
    )
    if updated.phase is GamePhase.PLAY and change.changed:
        updated = replace(
            updated,
            turn_counters=updated.turn_counters.increment(player_id, scored=1),
        )
    return updated, change


def return_card(
    state: GameState, card_id: CardId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    """Return a card to the bottom of its matching age supply."""

    age = registry.card(card_id).age
    return move_card(
        state,
        card_id,
        CardLocation.supply(age),
        registry,
        kind=ChangeKind.RETURN,
        placement=Placement.BOTTOM,
    )


def remove_card(
    state: GameState, card_id: CardId, registry: CardRegistry
) -> tuple[GameState, ChangeRecord]:
    """Set a card aside outside the game."""

    return move_card(
        state,
        card_id,
        CardLocation.removed(),
        registry,
        kind=ChangeKind.REMOVE,
    )


def exchange_cards(
    state: GameState,
    first_location: CardLocation,
    first_cards: tuple[CardId, ...],
    second_location: CardLocation,
    second_cards: tuple[CardId, ...],
    registry: CardRegistry,
) -> tuple[GameState, ChangeRecord]:
    """Atomically exchange selected cards, including when either side is empty."""

    _assert_mutable(state)
    if first_location == second_location:
        raise ZoneOperationError("an exchange requires two distinct locations")
    protected = {ZoneKind.NORMAL_ACHIEVEMENT, ZoneKind.REMOVED}
    if first_location.kind in protected or second_location.kind in protected:
        raise ZoneOperationError("achievements and removed cards cannot be exchanged")
    if len(set(first_cards + second_cards)) != len(first_cards) + len(second_cards):
        raise ZoneOperationError("exchange card selections overlap or contain duplicates")
    if any(card_id not in cards_at(state, first_location) for card_id in first_cards):
        raise ZoneOperationError("first exchange selection is not contained in its zone")
    if any(card_id not in cards_at(state, second_location) for card_id in second_cards):
        raise ZoneOperationError("second exchange selection is not contained in its zone")

    updated = state
    for card_id in first_cards:
        updated = _remove_from_zone(updated, first_location, card_id, registry)
    for card_id in second_cards:
        updated = _remove_from_zone(updated, second_location, card_id, registry)
    for card_id in second_cards:
        updated = _add_to_zone(updated, first_location, card_id, Placement.TOP, registry)
    for card_id in first_cards:
        updated = _add_to_zone(updated, second_location, card_id, Placement.TOP, registry)

    moves = tuple(
        [CardMove(card_id, first_location, second_location) for card_id in first_cards]
        + [CardMove(card_id, second_location, first_location) for card_id in second_cards]
    )
    assert_state_invariants(updated, registry)
    splay_changes = _movement_splay_changes(state, updated, (first_location, second_location))
    return updated, ChangeRecord(ChangeKind.EXCHANGE, moves, splay_changes)


def set_splay(
    state: GameState,
    player_id: PlayerId,
    color: Color,
    direction: SplayDirection,
    registry: CardRegistry,
) -> tuple[GameState, ChangeRecord]:
    """Set one stack's effective splay, emitting no change for a no-op."""

    _assert_mutable(state)
    player = state.player(player_id)
    before = player.board.stack(color)
    after = splay_stack(before, direction)
    if after.splay is before.splay:
        assert_state_invariants(state, registry)
        return state, ChangeRecord(ChangeKind.SPLAY)
    board = player.board.replace_stack(after)
    updated = state.replace_player(replace(player, board=board))
    assert_state_invariants(updated, registry)
    splay_change = SplayChange(player_id, color, before.splay, after.splay)
    return updated, ChangeRecord(ChangeKind.SPLAY, splay_changes=(splay_change,))


def rearrange_stack(
    state: GameState,
    player_id: PlayerId,
    color: Color,
    ordered_cards: tuple[CardId, ...],
    registry: CardRegistry,
) -> tuple[GameState, ChangeRecord]:
    """Reorder one stack bottom-to-top while retaining its previous splay."""

    _assert_mutable(state)
    location = CardLocation.board(player_id, color)
    existing = cards_at(state, location)
    if Counter(existing) != Counter(ordered_cards):
        raise ZoneOperationError("rearranged stack must contain exactly the existing cards")
    if existing == ordered_cards:
        assert_state_invariants(state, registry)
        return state, ChangeRecord(ChangeKind.REARRANGE)
    updated = _replace_zone_cards(state, location, ordered_cards, registry)
    assert_state_invariants(updated, registry)
    moves = tuple(CardMove(card_id, location, location) for card_id in ordered_cards)
    return updated, ChangeRecord(ChangeKind.REARRANGE, moves)
