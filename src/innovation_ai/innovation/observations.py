"""Detached player-safe observations under versioned information policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from innovation_ai.innovation.board import covered_visible_slots
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import GamePhase, GameState
from innovation_ai.innovation.types import (
    CardId,
    Color,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

OBSERVATION_SCHEMA_VERSION = 2


class InformationPolicy(StrEnum):
    """Supported policies for opponent covered-board information."""

    RULEBOOK_PRIVATE_COVERED = "rulebook-private-covered-v1"
    PUBLIC_COVERED = "public-covered-v1"


@dataclass(frozen=True, slots=True)
class ZoneObservation:
    """A hand or score pile: values are public, identities may be private.

    ``known_cards`` contains identities the viewer may legally see: everything in their own zone,
    plus any opponent card that is currently face up under a resolving instruction (rules
    decision 18).
    """

    values: tuple[int, ...]
    known_cards: tuple[CardId, ...]

    @property
    def count(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class CoveredCardObservation:
    """Information visible from one covered board card."""

    card_id: CardId | None
    age: int | None
    visible_icons: tuple[Icon, ...]


@dataclass(frozen=True, slots=True)
class StackObservation:
    """A board stack without leaking private covered identities or counts."""

    color: Color
    top_card_id: CardId | None
    splay: SplayDirection
    covered_cards: tuple[CoveredCardObservation, ...]
    covered_count: int | None


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """One player's public and viewer-relative private zones."""

    player_id: PlayerId
    hand: ZoneObservation
    score_pile: ZoneObservation
    board: tuple[StackObservation, ...]
    normal_achievements: tuple[NormalAchievementId, ...]
    special_achievements: tuple[SpecialAchievementId, ...]

    @property
    def achievement_count(self) -> int:
        return len(self.normal_achievements) + len(self.special_achievements)


@dataclass(frozen=True, slots=True)
class SupplyObservation:
    """Public count of one age supply; card order and identities remain hidden."""

    age: int
    count: int


@dataclass(frozen=True, slots=True)
class GameObservation:
    """Complete immutable observation supplied to one chooser."""

    viewer: PlayerId
    phase: GamePhase
    active_player: PlayerId | None
    turn_number: int
    paid_actions_remaining: int
    supplies: tuple[SupplyObservation, ...]
    players: tuple[PlayerObservation, PlayerObservation]
    available_normal_achievements: tuple[NormalAchievementId, ...]
    available_special_achievements: tuple[SpecialAchievementId, ...]
    information_policy: InformationPolicy
    rules_version: str
    revealed_cards: tuple[CardId, ...] = ()
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def player(self, player_id: PlayerId) -> PlayerObservation:
        """Return one observed player in canonical player order."""

        return self.players[tuple(PlayerId).index(player_id)]


def _zone_observation(
    card_ids: tuple[CardId, ...],
    *,
    reveal: bool,
    registry: CardRegistry,
    revealed: frozenset[CardId] = frozenset(),
) -> ZoneObservation:
    values = tuple(sorted(registry.card(card_id).age for card_id in card_ids))
    visible = card_ids if reveal else tuple(card_id for card_id in card_ids if card_id in revealed)
    return ZoneObservation(values, tuple(sorted(visible, key=str)))


def _stack_observation(
    state: GameState,
    owner: PlayerId,
    viewer: PlayerId,
    color: Color,
    policy: InformationPolicy,
    registry: CardRegistry,
) -> StackObservation:
    stack = state.player(owner).board.stack(color)
    top = stack.top
    covered = stack.cards[:-1]
    reveal_covered = owner is viewer or policy is InformationPolicy.PUBLIC_COVERED
    if reveal_covered:
        slots = covered_visible_slots(stack.splay)
        covered_cards = tuple(
            CoveredCardObservation(
                card_id,
                registry.card(card_id).age,
                tuple(
                    icon
                    for slot in slots
                    if (icon := registry.card(card_id).icon_at(slot)) is not None
                ),
            )
            for card_id in covered
        )
        covered_count: int | None = len(covered)
    elif stack.splay is SplayDirection.NONE:
        covered_cards = ()
        covered_count = None
    else:
        slots = covered_visible_slots(stack.splay)
        covered_cards = tuple(
            CoveredCardObservation(
                None,
                None,
                tuple(
                    icon
                    for slot in slots
                    if (icon := registry.card(card_id).icon_at(slot)) is not None
                ),
            )
            for card_id in covered
        )
        covered_count = len(covered)
    return StackObservation(color, top, stack.splay, covered_cards, covered_count)


def observe(
    state: GameState,
    viewer: PlayerId,
    registry: CardRegistry | None = None,
    *,
    policy: InformationPolicy | None = None,
) -> GameObservation:
    """Build a detached observation containing exactly information visible to ``viewer``."""

    registry = registry or load_card_registry()
    selected_policy = policy or InformationPolicy(state.information_policy_version)
    revealed = state.revealed_card_ids
    players = tuple(
        PlayerObservation(
            player.player_id,
            _zone_observation(
                player.hand,
                reveal=player.player_id is viewer,
                registry=registry,
                revealed=revealed,
            ),
            _zone_observation(
                player.score_pile,
                reveal=player.player_id is viewer,
                registry=registry,
                revealed=revealed,
            ),
            tuple(
                _stack_observation(
                    state, player.player_id, viewer, color, selected_policy, registry
                )
                for color in Color
            ),
            player.normal_achievements,
            player.special_achievements,
        )
        for player in state.players
    )
    claimed_normal = {
        achievement for player in state.players for achievement in player.normal_achievements
    }
    claimed_special = {
        achievement for player in state.players for achievement in player.special_achievements
    }
    return GameObservation(
        viewer=viewer,
        phase=state.phase,
        active_player=state.active_player,
        turn_number=state.turn_number,
        paid_actions_remaining=state.paid_actions_remaining,
        supplies=tuple(SupplyObservation(age, len(state.supply.pile(age))) for age in range(1, 11)),
        players=players,  # type: ignore[arg-type]
        available_normal_achievements=tuple(
            item for item in NormalAchievementId if item not in claimed_normal
        ),
        available_special_achievements=tuple(
            item for item in SpecialAchievementId if item not in claimed_special
        ),
        information_policy=selected_policy,
        rules_version=state.rules_version,
        revealed_cards=tuple(sorted(revealed, key=str)),
    )
