"""Shared deterministic board fixtures for the WP6 achievement and terminal suites."""

from __future__ import annotations

from dataclasses import replace

from innovation_ai.innovation.achievements import MonumentCountKind
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TurnCounters,
    build_setup_state_from_piles,
)
from innovation_ai.innovation.types import (
    CardId,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import meld_card, score_card, set_splay

ACTIVE = PlayerId.PLAYER_1
OPPONENT = PlayerId.PLAYER_2

# The deterministic explicit setup removes the alphabetically first card of ages 1-9 as that
# age's hidden normal achievement, so those identities are never available for board fixtures.
RESERVED_ACHIEVEMENT_CARDS = frozenset(
    {
        "agriculture",
        "calendar",
        "alchemy",
        "anatomy",
        "astronomy",
        "atomic-theory",
        "bicycle",
        "antibiotics",
        "collaboration",
    }
)

EMPIRE_SINGLES = (("tools",), ("currency",), ("gunpowder",), ("refrigeration",))
EMPIRE_SPLAYED = ("railroad", "the-internet", "feudalism")
WORLD_BLUE = ("quantum-theory", "rocketry", "software")
WORLD_GREEN = ("satellites", "databases")
# Each fixture provides two disjoint card sets so both players can hold the same board shape.
UNIVERSE_TOPS = (
    ("computers", "corporations", "empiricism", "flight", "skyscrapers"),
    ("genetics", "mass-media", "socialism", "mobility", "suburbia"),
)
WONDER_STACKS = (
    (
        (("pottery", "tools"), SplayDirection.RIGHT),
        (("the-wheel", "clothing"), SplayDirection.RIGHT),
        (("city-states", "code-of-laws"), SplayDirection.UP),
        (("archery", "oars"), SplayDirection.RIGHT),
        (("masonry", "domestication"), SplayDirection.UP),
    ),
    (
        (("writing", "mathematics"), SplayDirection.RIGHT),
        (("sailing", "currency"), SplayDirection.UP),
        (("mysticism", "monotheism"), SplayDirection.RIGHT),
        (("metalworking", "construction"), SplayDirection.UP),
        (("fermenting", "canal-building"), SplayDirection.RIGHT),
    ),
)
FIVE_NORMAL_ACHIEVEMENTS = (
    NormalAchievementId.AGE_1,
    NormalAchievementId.AGE_2,
    NormalAchievementId.AGE_3,
    NormalAchievementId.AGE_4,
    NormalAchievementId.AGE_5,
)
FOUR_NORMAL_ACHIEVEMENTS = (
    NormalAchievementId.AGE_6,
    NormalAchievementId.AGE_7,
    NormalAchievementId.AGE_8,
    NormalAchievementId.AGE_9,
)


def card_registry() -> CardRegistry:
    """Return the cached packaged card registry."""

    return load_card_registry()


def playable_state(registry: CardRegistry, *, active: PlayerId = ACTIVE) -> GameState:
    """Return a deterministic mid-play state with empty boards and empty score piles."""

    piles = tuple(
        tuple(sorted((card.id for card in registry.cards if card.age == age), key=str))
        for age in range(1, 11)
    )
    state = build_setup_state_from_piles(piles, seed=0, registry=registry)
    return replace(
        state,
        phase=GamePhase.PLAY,
        active_player=active,
        turn_number=3,
        paid_actions_remaining=2,
        starting_meld_choices=(None, None),
    )


def place(
    state: GameState,
    player_id: PlayerId,
    names: tuple[str, ...],
    registry: CardRegistry,
    *,
    splay: SplayDirection = SplayDirection.NONE,
) -> GameState:
    """Meld ``names`` bottom-to-top onto one board and optionally splay that color."""

    for name in names:
        assert name not in RESERVED_ACHIEVEMENT_CARDS
        state, _ = meld_card(state, player_id, CardId(name), registry)
    if splay is not SplayDirection.NONE:
        color = registry.card(CardId(names[0])).color
        state, _ = set_splay(state, player_id, color, splay, registry)
    return state


def with_score(
    state: GameState, player_id: PlayerId, ages: tuple[int, ...], registry: CardRegistry
) -> GameState:
    """Score one supply card of each requested age into ``player_id``'s score pile."""

    for age in ages:
        state, _ = score_card(state, player_id, state.supply.pile(age)[-1], registry)
    # Fixture scoring must not leave Monument progress behind.
    return replace(state, turn_counters=TurnCounters.empty())


def with_achievements(
    state: GameState,
    player_id: PlayerId,
    normal: tuple[NormalAchievementId, ...] = (),
    special: tuple[SpecialAchievementId, ...] = (),
) -> GameState:
    player = state.player(player_id)
    return state.replace_player(
        replace(
            player,
            normal_achievements=(*player.normal_achievements, *normal),
            special_achievements=(*player.special_achievements, *special),
        )
    )


def empire_board(state: GameState, player_id: PlayerId, registry: CardRegistry) -> GameState:
    for names in EMPIRE_SINGLES:
        state = place(state, player_id, names, registry)
    return place(state, player_id, EMPIRE_SPLAYED, registry, splay=SplayDirection.UP)


def world_board(state: GameState, player_id: PlayerId, registry: CardRegistry) -> GameState:
    state = place(state, player_id, WORLD_BLUE, registry, splay=SplayDirection.UP)
    return place(state, player_id, WORLD_GREEN, registry, splay=SplayDirection.UP)


def universe_board(
    state: GameState, player_id: PlayerId, registry: CardRegistry, *, variant: int = 0
) -> GameState:
    for name in UNIVERSE_TOPS[variant]:
        state = place(state, player_id, (name,), registry)
    return state


def wonder_board(
    state: GameState, player_id: PlayerId, registry: CardRegistry, *, variant: int = 0
) -> GameState:
    for names, splay in WONDER_STACKS[variant]:
        state = place(state, player_id, names, registry, splay=splay)
    return state


def monument_counters(
    state: GameState, player_id: PlayerId, kind: MonumentCountKind, count: int
) -> GameState:
    counters = state.turn_counters
    for _ in range(count):
        if kind is MonumentCountKind.TUCK:
            counters = counters.increment(player_id, tucked=1)
        else:
            counters = counters.increment(player_id, scored=1)
    return replace(state, turn_counters=counters)
