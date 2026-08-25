"""Normal and special achievement legality, claiming, and atomic-boundary checks.

This module owns every achievement rule and is the single entry point that other work packages
call when an atomic operation completes.

Integration contract for the effect and dogma work packages (WP4/WP5)
--------------------------------------------------------------------
The entry points below are deliberately generic: they take authoritative state plus optional
plain-data hints, never effect frames, effect contexts, or provenance records owned by WP4. That
keeps this module compilable and testable while provenance types are still being designed.

* :func:`check_atomic_boundary` is the hook to call after **every** atomic operation, after each
  completed dogma effect, after each paid action, and at turn completion. It claims every
  now-eligible special achievement in the deterministic order fixed by
  ``docs/RULES_DECISIONS.md`` decision 3 and returns a terminal result as soon as a sixth
  achievement is claimed. When it returns a terminal result, the caller must abandon all
  remaining dogma work, including the sharing bonus.
* :func:`check_after_change` is the same hook for callers holding a WP2 :class:`ChangeRecord`;
  it optionally folds Monument counters for movement performed outside the WP2 counting
  primitives before running the boundary check.
* :func:`claim_linked_route` implements the five linked-card alternate routes. Card effects call
  it instead of duplicating claim bookkeeping.
* :func:`record_qualifying_movements` is the provenance-friendly Monument counter API. WP2's
  :func:`~innovation_ai.innovation.zones.tuck_card` and
  :func:`~innovation_ai.innovation.zones.score_card` already count their own single-card
  movements, so this function is only for bulk atoms written without those primitives. Transfers
  and exchanges never count, which :func:`qualifying_monument_movements` enforces by change kind.

When WP4's provenance record exists, the expected integration is a thin adapter owned by WP4
that maps one provenance batch to ``tuple[QualifyingMovement, ...]`` plus the executing player
and then calls the functions above. No signature in this module needs to change.

Per ``docs/RULES_DECISIONS.md`` decision 2, an achievement claim is a player-facing gameplay
change and therefore qualifies an outer shared execution for the sharing bonus. The effect VM
emits explicit achievement events from :attr:`AchievementCheckResult.claims`; this module keeps
claim bookkeeping independent of effect provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from innovation_ai.innovation.board import (
    highest_top_value,
    score_value,
    top_cards,
    visible_icons,
)
from innovation_ai.innovation.catalog import (
    CardRegistry,
    LinkedAchievementRoute,
    load_card_registry,
)
from innovation_ai.innovation.state import GamePhase, GameState, TerminalResult
from innovation_ai.innovation.terminal import (
    achievement_victory_result,
    apply_terminal,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import (
    ChangeKind,
    ChangeRecord,
    ZoneKind,
    assert_state_invariants,
)

EMPIRE_ICONS_PER_TYPE = 3
WORLD_CLOCK_COUNT = 12
UNIVERSE_MINIMUM_TOP_VALUE = 8
MONUMENT_MOVEMENT_COUNT = 6
MASONRY_MELD_COUNT = 4
ASTRONOMY_MINIMUM_TOP_VALUE = 6
NORMAL_ACHIEVEMENT_SCORE_MULTIPLIER = 5

SPECIAL_CHECK_ORDER: tuple[SpecialAchievementId, ...] = (
    SpecialAchievementId.MONUMENT,
    SpecialAchievementId.EMPIRE,
    SpecialAchievementId.WORLD,
    SpecialAchievementId.WONDER,
    SpecialAchievementId.UNIVERSE,
)
"""Per-player special-achievement check order from ``docs/RULES_DECISIONS.md`` decision 3."""


class AchievementClaimError(ValueError):
    """A requested achievement claim is not currently legal."""


class ClaimRoute(StrEnum):
    """How an achievement came to be claimed."""

    ACHIEVE_ACTION = "achieve-action"
    AUTOMATIC = "automatic"
    LINKED_CARD = "linked-card"


class MonumentCountKind(StrEnum):
    """The two separately counted Monument conditions."""

    TUCK = "tuck"
    SCORE = "score"


_MONUMENT_COUNTING_CHANGE_KINDS: dict[ChangeKind, MonumentCountKind] = {
    ChangeKind.TUCK: MonumentCountKind.TUCK,
    ChangeKind.SCORE: MonumentCountKind.SCORE,
}
"""Only tuck and score change kinds count; transfers and exchanges never do."""


@dataclass(frozen=True, slots=True)
class QualifyingMovement:
    """One card movement that counts toward Monument for exactly one player."""

    player_id: PlayerId
    kind: MonumentCountKind
    card_id: CardId | None = None


@dataclass(frozen=True, slots=True)
class AchievementClaim:
    """A single achievement that changed owner during one boundary check."""

    player_id: PlayerId
    achievement_id: NormalAchievementId | SpecialAchievementId
    route: ClaimRoute
    source_effect_id: DogmaEffectId | None = None

    def __post_init__(self) -> None:
        if self.route is ClaimRoute.LINKED_CARD and self.source_effect_id is None:
            raise ValueError("a linked-card claim must record its source effect")
        if self.route is not ClaimRoute.LINKED_CARD and self.source_effect_id is not None:
            raise ValueError("only a linked-card claim has a source effect")


@dataclass(frozen=True, slots=True)
class AchievementCheckResult:
    """Outcome of one achievement boundary check or explicit claim."""

    state: GameState
    claims: tuple[AchievementClaim, ...] = ()
    terminal: TerminalResult | None = None

    def __post_init__(self) -> None:
        if (self.terminal is not None) != (self.state.phase is GamePhase.TERMINAL):
            raise ValueError("terminal result and state phase disagree")

    @property
    def game_over(self) -> bool:
        """Whether the caller must abandon all remaining effect and turn work."""

        return self.terminal is not None

    @property
    def changed(self) -> bool:
        """Whether any achievement was claimed during this check."""

        return bool(self.claims)


@dataclass(frozen=True, slots=True)
class LinkedRouteContext:
    """Effect-scoped inputs a linked-card route needs beyond authoritative state.

    ``melded_card_count`` carries Masonry's "if you melded four or more cards in this way"
    tally. State-based routes ignore it.
    """

    melded_card_count: int = 0

    def __post_init__(self) -> None:
        if self.melded_card_count < 0:
            raise ValueError("melded card count cannot be negative")


_EMPTY_LINKED_CONTEXT = LinkedRouteContext()


class _LinkedRoutePredicate(Protocol):
    """Signature shared by every linked-card alternate-route predicate."""

    def __call__(
        self,
        state: GameState,
        player_id: PlayerId,
        registry: CardRegistry,
        *,
        context: LinkedRouteContext,
    ) -> bool: ...


def normal_achievement_age(achievement_id: NormalAchievementId) -> int:
    """Return the age 1-9 of a normal achievement."""

    return tuple(NormalAchievementId).index(achievement_id) + 1


def claimed_normal_achievements(state: GameState) -> frozenset[NormalAchievementId]:
    """Return every normal achievement already owned by either player."""

    return frozenset(
        achievement for player in state.players for achievement in player.normal_achievements
    )


def claimed_special_achievements(state: GameState) -> frozenset[SpecialAchievementId]:
    """Return every special achievement already owned by either player."""

    return frozenset(
        achievement for player in state.players for achievement in player.special_achievements
    )


def available_normal_achievements(state: GameState) -> tuple[NormalAchievementId, ...]:
    """Return unclaimed normal achievements in canonical age order."""

    claimed = claimed_normal_achievements(state)
    return tuple(item for item in NormalAchievementId if item not in claimed)


def available_special_achievements(state: GameState) -> tuple[SpecialAchievementId, ...]:
    """Return unclaimed special achievements in canonical enum order."""

    claimed = claimed_special_achievements(state)
    return tuple(item for item in SpecialAchievementId if item not in claimed)


def normal_achievement_is_eligible(
    state: GameState,
    player_id: PlayerId,
    achievement_id: NormalAchievementId,
    registry: CardRegistry | None = None,
) -> bool:
    """Whether ``player_id`` may claim one specific normal achievement right now.

    Both printed conditions must hold: score at least five times the age, and at least one top
    card whose value is that age or higher.
    """

    registry = registry or load_card_registry()
    if achievement_id in claimed_normal_achievements(state):
        return False
    player = state.player(player_id)
    age = normal_achievement_age(achievement_id)
    if score_value(player, registry) < NORMAL_ACHIEVEMENT_SCORE_MULTIPLIER * age:
        return False
    return highest_top_value(player.board, registry) >= age


def eligible_normal_achievements(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> tuple[NormalAchievementId, ...]:
    """Return every normal achievement ``player_id`` may claim, in canonical age order."""

    registry = registry or load_card_registry()
    return tuple(
        achievement_id
        for achievement_id in NormalAchievementId
        if normal_achievement_is_eligible(state, player_id, achievement_id, registry)
    )


def _award(
    state: GameState,
    player_id: PlayerId,
    achievement_id: NormalAchievementId | SpecialAchievementId,
    registry: CardRegistry,
) -> GameState:
    player = state.player(player_id)
    if isinstance(achievement_id, NormalAchievementId):
        replacement = replace(
            player, normal_achievements=(*player.normal_achievements, achievement_id)
        )
    else:
        replacement = replace(
            player, special_achievements=(*player.special_achievements, achievement_id)
        )
    updated = state.replace_player(replacement)
    assert_state_invariants(updated, registry)
    return updated


def _claim(
    state: GameState,
    player_id: PlayerId,
    achievement_id: NormalAchievementId | SpecialAchievementId,
    route: ClaimRoute,
    registry: CardRegistry,
    *,
    source_effect_id: DogmaEffectId | None = None,
    previous_claims: tuple[AchievementClaim, ...] = (),
) -> AchievementCheckResult:
    """Award one achievement and immediately resolve the sixth-achievement victory."""

    updated = _award(state, player_id, achievement_id, registry)
    claims = (
        *previous_claims,
        AchievementClaim(player_id, achievement_id, route, source_effect_id),
    )
    victory = achievement_victory_result(updated, player_id)
    if victory is not None:
        return AchievementCheckResult(apply_terminal(updated, victory), claims, victory)
    return AchievementCheckResult(updated, claims)


def claim_normal_achievement(
    state: GameState,
    player_id: PlayerId,
    achievement_id: NormalAchievementId,
    registry: CardRegistry | None = None,
    *,
    route: ClaimRoute = ClaimRoute.ACHIEVE_ACTION,
) -> AchievementCheckResult:
    """Claim an eligible normal achievement, ending the game on a sixth achievement."""

    registry = registry or load_card_registry()
    if not normal_achievement_is_eligible(state, player_id, achievement_id, registry):
        raise AchievementClaimError(f"{player_id} cannot claim {achievement_id}")
    return _claim(state, player_id, achievement_id, route, registry)


# --------------------------------------------------------------------------------------------
# Monument counters
# --------------------------------------------------------------------------------------------


def qualifying_monument_movements(change: ChangeRecord) -> tuple[QualifyingMovement, ...]:
    """Classify a WP2 change record into Monument-qualifying movements.

    Only ``tuck`` and ``score`` change kinds qualify. Transfers into a score pile and cards
    entering a score pile through an exchange are excluded by the printed Monument rule, and
    that exclusion is expressed here purely by change kind so no card-specific code can bypass
    it.
    """

    counted = _MONUMENT_COUNTING_CHANGE_KINDS.get(change.kind)
    if counted is None:
        return ()
    expected_zone = ZoneKind.BOARD if counted is MonumentCountKind.TUCK else ZoneKind.SCORE
    movements: list[QualifyingMovement] = []
    for move in change.card_moves:
        destination = move.destination
        if destination.kind is not expected_zone or destination.player_id is None:
            continue
        movements.append(QualifyingMovement(destination.player_id, counted, move.card_id))
    return tuple(movements)


def record_qualifying_movements(
    state: GameState, movements: Iterable[QualifyingMovement]
) -> GameState:
    """Fold Monument-qualifying movements into the per-turn counters.

    WP2's ``tuck_card`` and ``score_card`` primitives already count their own movement, so this
    is only for bulk atoms that write zones without them. Counters advance only during play, so
    setup melds never contribute.
    """

    if state.phase is not GamePhase.PLAY:
        return state
    counters = state.turn_counters
    changed = False
    for movement in movements:
        changed = True
        if movement.kind is MonumentCountKind.TUCK:
            counters = counters.increment(movement.player_id, tucked=1)
        else:
            counters = counters.increment(movement.player_id, scored=1)
    return replace(state, turn_counters=counters) if changed else state


def monument_progress(state: GameState, player_id: PlayerId) -> tuple[int, int]:
    """Return ``(tucked, scored)`` qualifying counts for this turn."""

    counters = state.turn_counters.for_player(player_id)
    return counters.tucked, counters.scored


# --------------------------------------------------------------------------------------------
# Automatic special-achievement predicates
#
# Every predicate shares the ``(state, player_id, registry)`` signature so they can be dispatched
# uniformly, even when a particular predicate does not need the registry.
# --------------------------------------------------------------------------------------------


def monument_predicate(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> bool:
    """Tucked at least six or scored at least six cards during the current turn.

    The two conditions are separate and are never added together.
    """

    tucked, scored = monument_progress(state, player_id)
    return tucked >= MONUMENT_MOVEMENT_COUNT or scored >= MONUMENT_MOVEMENT_COUNT


def empire_predicate(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> bool:
    """At least three visible icons of every one of the six icon types."""

    registry = registry or load_card_registry()
    icons = visible_icons(state.player(player_id).board, registry)
    return all(icons[icon] >= EMPIRE_ICONS_PER_TYPE for icon in Icon)


def world_predicate(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> bool:
    """At least twelve visible clock icons."""

    registry = registry or load_card_registry()
    return visible_icons(state.player(player_id).board, registry)[Icon.CLOCK] >= WORLD_CLOCK_COUNT


def wonder_predicate(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> bool:
    """All five colors present and each splayed right or up."""

    return all(
        stack.cards and stack.splay in {SplayDirection.RIGHT, SplayDirection.UP}
        for stack in state.player(player_id).board.stacks
    )


def universe_predicate(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> bool:
    """Five top cards, one of each color, each of value eight or higher."""

    registry = registry or load_card_registry()
    board = state.player(player_id).board
    tops = top_cards(board)
    if len(tops) != len(Color):
        return False
    return all(registry.card(card_id).age >= UNIVERSE_MINIMUM_TOP_VALUE for card_id in tops)


_AUTOMATIC_PREDICATES: dict[
    SpecialAchievementId, Callable[[GameState, PlayerId, CardRegistry], bool]
] = {
    SpecialAchievementId.MONUMENT: monument_predicate,
    SpecialAchievementId.EMPIRE: empire_predicate,
    SpecialAchievementId.WORLD: world_predicate,
    SpecialAchievementId.WONDER: wonder_predicate,
    SpecialAchievementId.UNIVERSE: universe_predicate,
}


def automatic_predicate_satisfied(
    state: GameState,
    player_id: PlayerId,
    achievement_id: SpecialAchievementId,
    registry: CardRegistry | None = None,
) -> bool:
    """Evaluate one automatic special-achievement predicate against live state."""

    registry = registry or load_card_registry()
    return _AUTOMATIC_PREDICATES[achievement_id](state, player_id, registry)


def automatically_eligible_special_achievements(
    state: GameState, player_id: PlayerId, registry: CardRegistry | None = None
) -> tuple[SpecialAchievementId, ...]:
    """Return unclaimed special achievements whose automatic predicate now holds.

    The result is ordered by :data:`SPECIAL_CHECK_ORDER`.
    """

    registry = registry or load_card_registry()
    claimed = claimed_special_achievements(state)
    return tuple(
        achievement_id
        for achievement_id in SPECIAL_CHECK_ORDER
        if achievement_id not in claimed
        and automatic_predicate_satisfied(state, player_id, achievement_id, registry)
    )


# --------------------------------------------------------------------------------------------
# Linked-card alternate routes
#
# Every route shares the ``(state, player_id, registry, *, context)`` signature for uniform
# dispatch. Routes deliberately differ from the automatic predicates above.
# --------------------------------------------------------------------------------------------


def masonry_monument_route(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Masonry: melded four or more castle cards from hand during this effect."""

    return context.melded_card_count >= MASONRY_MELD_COUNT


def construction_empire_route(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Construction: you are the only player with five top cards."""

    if len(top_cards(state.player(player_id).board)) != len(Color):
        return False
    return all(
        len(top_cards(player.board)) != len(Color)
        for player in state.players
        if player.player_id is not player_id
    )


def translation_world_route(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Translation: every top card on your board has a crown.

    Per ``docs/RULES_DECISIONS.md`` decision 10, the universal predicate is vacuously true for
    an empty board.
    """

    registry = registry or load_card_registry()
    return all(
        Icon.CROWN in registry.card(card_id).functional_icons
        for card_id in top_cards(state.player(player_id).board)
    )


def invention_wonder_route(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Invention: five colors splayed, each in any direction."""

    return all(
        stack.splay is not SplayDirection.NONE for stack in state.player(player_id).board.stacks
    )


def astronomy_universe_route(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Astronomy: all non-purple top cards are value six or higher.

    Per ``docs/RULES_DECISIONS.md`` decision 10, an empty non-purple top-card set satisfies the
    condition.
    """

    registry = registry or load_card_registry()
    board = state.player(player_id).board
    return all(
        registry.card(card_id).age >= ASTRONOMY_MINIMUM_TOP_VALUE
        for stack in board.stacks
        if stack.color is not Color.PURPLE and (card_id := stack.top) is not None
    )


_LINKED_ROUTES: dict[SpecialAchievementId, _LinkedRoutePredicate] = {
    SpecialAchievementId.MONUMENT: masonry_monument_route,
    SpecialAchievementId.EMPIRE: construction_empire_route,
    SpecialAchievementId.WORLD: translation_world_route,
    SpecialAchievementId.WONDER: invention_wonder_route,
    SpecialAchievementId.UNIVERSE: astronomy_universe_route,
}


def linked_route(
    achievement_id: SpecialAchievementId, registry: CardRegistry | None = None
) -> LinkedAchievementRoute:
    """Return the card effect that grants the alternate route to ``achievement_id``."""

    registry = registry or load_card_registry()
    return registry.linked_achievement_routes[achievement_id]


def linked_route_satisfied(
    state: GameState,
    player_id: PlayerId,
    achievement_id: SpecialAchievementId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
) -> bool:
    """Evaluate the linked-card route predicate, which differs from the automatic one."""

    registry = registry or load_card_registry()
    return _LINKED_ROUTES[achievement_id](state, player_id, registry, context=context)


def claim_linked_route(
    state: GameState,
    player_id: PlayerId,
    achievement_id: SpecialAchievementId,
    registry: CardRegistry | None = None,
    *,
    context: LinkedRouteContext = _EMPTY_LINKED_CONTEXT,
    check_boundary: bool = True,
) -> AchievementCheckResult:
    """Claim a special achievement through its linked card's alternate route.

    Returns an unchanged result when the route condition fails or the achievement is already
    owned; the linked effect then simply does nothing. When ``check_boundary`` is true the
    standard atomic boundary check runs afterwards so any other now-eligible achievement and
    the sixth-achievement victory resolve in the documented order.
    """

    registry = registry or load_card_registry()
    if achievement_id in claimed_special_achievements(state):
        return AchievementCheckResult(state)
    if not linked_route_satisfied(state, player_id, achievement_id, registry, context=context):
        return AchievementCheckResult(state)
    route = linked_route(achievement_id, registry)
    result = _claim(
        state,
        player_id,
        achievement_id,
        ClaimRoute.LINKED_CARD,
        registry,
        source_effect_id=route.source_effect_id,
    )
    if result.game_over or not check_boundary:
        return result
    return check_atomic_boundary(result.state, registry, previous_claims=result.claims)


# --------------------------------------------------------------------------------------------
# Generic atomic-boundary entry points
# --------------------------------------------------------------------------------------------


def check_order(
    state: GameState, active_player: PlayerId | None = None
) -> tuple[tuple[PlayerId, SpecialAchievementId], ...]:
    """Return the deterministic (player, achievement) evaluation order for a boundary.

    The active player is checked completely before the opponent, and each player's achievements
    follow :data:`SPECIAL_CHECK_ORDER`. This satisfies the rulebook's same-achievement
    active-player priority and extends it to a total order for engine determinism.
    """

    first = active_player if active_player is not None else state.active_player
    if first is None:
        players = tuple(PlayerId)
    else:
        players = (first, *(player_id for player_id in PlayerId if player_id is not first))
    return tuple(
        (player_id, achievement_id)
        for player_id in players
        for achievement_id in SPECIAL_CHECK_ORDER
    )


def check_atomic_boundary(
    state: GameState,
    registry: CardRegistry | None = None,
    *,
    active_player: PlayerId | None = None,
    previous_claims: tuple[AchievementClaim, ...] = (),
) -> AchievementCheckResult:
    """Claim every automatically eligible special achievement at an atomic boundary.

    This is the generic hook for the effect and dogma work packages: call it after every atomic
    operation, completed effect, paid action, and turn. Predicates use live state rather than
    frozen dogma icon counts. As soon as a claim produces a sixth achievement the returned
    result carries a terminal result and the caller must stop all remaining work, including the
    sharing bonus.
    """

    registry = registry or load_card_registry()
    if state.phase is GamePhase.TERMINAL:
        assert state.terminal_result is not None
        return AchievementCheckResult(state, previous_claims, state.terminal_result)

    current = state
    claims = previous_claims
    for player_id, achievement_id in check_order(state, active_player):
        if achievement_id in claimed_special_achievements(current):
            continue
        if not automatic_predicate_satisfied(current, player_id, achievement_id, registry):
            continue
        result = _claim(
            current,
            player_id,
            achievement_id,
            ClaimRoute.AUTOMATIC,
            registry,
            previous_claims=claims,
        )
        current, claims = result.state, result.claims
        if result.game_over:
            return result
    return AchievementCheckResult(current, claims)


def check_after_change(
    state: GameState,
    change: ChangeRecord | None = None,
    registry: CardRegistry | None = None,
    *,
    active_player: PlayerId | None = None,
    count_monument_movements: bool = False,
) -> AchievementCheckResult:
    """Boundary check for callers holding a WP2 change record.

    Set ``count_monument_movements`` only when the movement was performed without WP2's
    counting ``tuck_card``/``score_card`` primitives; otherwise counters would double count.
    """

    registry = registry or load_card_registry()
    current = state
    if change is not None and count_monument_movements:
        current = record_qualifying_movements(current, qualifying_monument_movements(change))
    return check_atomic_boundary(current, registry, active_player=active_player)
