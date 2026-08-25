"""Authoritative immutable state for a two-player Innovation game."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum

from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

RULES_VERSION = "innovation-base-third-edition-2p-v1"
INFORMATION_POLICY_VERSION = "rulebook-private-covered-v1"
STATE_SCHEMA_VERSION = 1
TERMINAL_SCHEMA_VERSION = 1
SETUP_RNG_VERSION = "python-mt19937-shuffle-v1"

type StateScalar = str | int | bool | None
type StateValue = StateScalar | tuple[StateValue, ...]


class TerminalReason(StrEnum):
    """Stable reasons why an Innovation game ended."""

    ACHIEVEMENT_VICTORY = "achievement-victory"
    DRAW_BEYOND_AGE_10 = "draw-beyond-age-10"
    CARD_EFFECT = "card-effect"


class GamePhase(StrEnum):
    """Coarse phase of the authoritative state machine."""

    STARTING_MELDS = "starting-melds"
    PLAY = "play"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ColorStack:
    """One color stack, ordered from bottom card to top card."""

    color: Color
    cards: tuple[CardId, ...] = ()
    splay: SplayDirection = SplayDirection.NONE

    def __post_init__(self) -> None:
        if len(self.cards) <= 1 and self.splay is not SplayDirection.NONE:
            raise ValueError("a zero- or one-card stack must be unsplayed")
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("a color stack cannot contain a card more than once")

    @property
    def top(self) -> CardId | None:
        """Return the completely visible top card."""

        return self.cards[-1] if self.cards else None

    @property
    def bottom(self) -> CardId | None:
        """Return the bottom card."""

        return self.cards[0] if self.cards else None

    @property
    def beneath_top(self) -> CardId | None:
        """Return the card immediately beneath the top card, if any."""

        return self.cards[-2] if len(self.cards) >= 2 else None


@dataclass(frozen=True, slots=True)
class Board:
    """A player's five color stacks in canonical color order."""

    stacks: tuple[ColorStack, ...]

    def __post_init__(self) -> None:
        if tuple(stack.color for stack in self.stacks) != tuple(Color):
            raise ValueError("board must contain exactly one stack in canonical color order")

    @classmethod
    def empty(cls) -> Board:
        """Construct an empty five-color board."""

        return cls(tuple(ColorStack(color) for color in Color))

    def stack(self, color: Color) -> ColorStack:
        """Return the stack for ``color``."""

        return self.stacks[tuple(Color).index(color)]

    def replace_stack(self, replacement: ColorStack) -> Board:
        """Return a board with one color stack replaced."""

        return Board(
            tuple(
                replacement if stack.color is replacement.color else stack for stack in self.stacks
            )
        )


@dataclass(frozen=True, slots=True)
class PlayerState:
    """All authoritative zones and achievements belonging to one player."""

    player_id: PlayerId
    hand: tuple[CardId, ...]
    board: Board
    score_pile: tuple[CardId, ...]
    normal_achievements: tuple[NormalAchievementId, ...] = ()
    special_achievements: tuple[SpecialAchievementId, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.hand)) != len(self.hand):
            raise ValueError("a hand cannot contain a card more than once")
        if len(set(self.score_pile)) != len(self.score_pile):
            raise ValueError("a score pile cannot contain a card more than once")
        if len(set(self.normal_achievements)) != len(self.normal_achievements):
            raise ValueError("normal achievements cannot be duplicated")
        if len(set(self.special_achievements)) != len(self.special_achievements):
            raise ValueError("special achievements cannot be duplicated")
        object.__setattr__(self, "hand", tuple(sorted(self.hand, key=str)))
        object.__setattr__(self, "score_pile", tuple(sorted(self.score_pile, key=str)))
        object.__setattr__(
            self,
            "normal_achievements",
            tuple(sorted(self.normal_achievements, key=lambda item: item.value)),
        )
        object.__setattr__(
            self,
            "special_achievements",
            tuple(sorted(self.special_achievements, key=lambda item: item.value)),
        )

    @property
    def achievement_count(self) -> int:
        """Return the player's total claimed achievement count."""

        return len(self.normal_achievements) + len(self.special_achievements)


@dataclass(frozen=True, slots=True)
class SupplyState:
    """Ordered age supplies; each pile is ordered top to bottom."""

    piles: tuple[tuple[CardId, ...], ...]

    def __post_init__(self) -> None:
        if len(self.piles) != 10:
            raise ValueError("supply must contain exactly ten age piles")

    def pile(self, age: int) -> tuple[CardId, ...]:
        """Return an age pile ordered top to bottom."""

        if not 1 <= age <= 10:
            raise ValueError(f"supply age must be 1-10, got {age}")
        return self.piles[age - 1]

    def replace_pile(self, age: int, cards: tuple[CardId, ...]) -> SupplyState:
        """Return supplies with one age pile replaced."""

        piles = list(self.piles)
        piles[age - 1] = cards
        return SupplyState(tuple(piles))


@dataclass(frozen=True, slots=True)
class NormalAchievementState:
    """Hidden identities of the nine normal achievements."""

    cards: tuple[CardId, ...]

    def __post_init__(self) -> None:
        if len(self.cards) != 9:
            raise ValueError("there must be one normal-achievement card for ages 1-9")

    def card(self, achievement_id: NormalAchievementId) -> CardId:
        """Return the hidden identity for internal engine use only."""

        return self.cards[tuple(NormalAchievementId).index(achievement_id)]


@dataclass(frozen=True, slots=True)
class PlayerTurnCounters:
    """Per-player qualifying movement counts during the current turn."""

    player_id: PlayerId
    tucked: int = 0
    scored: int = 0

    def __post_init__(self) -> None:
        if self.tucked < 0 or self.scored < 0:
            raise ValueError("turn counters cannot be negative")


@dataclass(frozen=True, slots=True)
class TurnCounters:
    """Counters whose lifetime is one paid turn."""

    players: tuple[PlayerTurnCounters, PlayerTurnCounters]

    def __post_init__(self) -> None:
        if tuple(counter.player_id for counter in self.players) != tuple(PlayerId):
            raise ValueError("turn counters must be in canonical player order")

    @classmethod
    def empty(cls) -> TurnCounters:
        """Construct zeroed counters for both players."""

        return cls(tuple(PlayerTurnCounters(player) for player in PlayerId))  # type: ignore[arg-type]

    def for_player(self, player_id: PlayerId) -> PlayerTurnCounters:
        """Return counters for one player."""

        return self.players[tuple(PlayerId).index(player_id)]

    def increment(self, player_id: PlayerId, *, tucked: int = 0, scored: int = 0) -> TurnCounters:
        """Return counters with qualifying movement added for one player."""

        if tucked < 0 or scored < 0:
            raise ValueError("counter increments cannot be negative")
        current = self.for_player(player_id)
        replacement = PlayerTurnCounters(
            player_id,
            tucked=current.tucked + tucked,
            scored=current.scored + scored,
        )
        players = tuple(
            replacement if counter.player_id is player_id else counter for counter in self.players
        )
        return TurnCounters(players)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EffectVariable:
    """One serializable variable scoped to pending effect resolution."""

    name: str
    value: StateValue

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("effect variable name cannot be empty")


@dataclass(frozen=True, slots=True)
class EffectFrameState:
    """Serializable placeholder frame owned by later effect work packages."""

    kind: str
    step: int = 0
    source_card_id: CardId | None = None
    variables: tuple[EffectVariable, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("effect frame kind cannot be empty")
        if self.step < 0:
            raise ValueError("effect frame step cannot be negative")


@dataclass(frozen=True, slots=True)
class SetupProvenance:
    """Random setup inputs and explicit shuffled order for exact replay."""

    seed: int
    card_data_fingerprint: str
    shuffled_piles: tuple[tuple[CardId, ...], ...]
    deal_sequence: tuple[PlayerId, ...]
    rng_version: str = SETUP_RNG_VERSION


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Typed serializable terminal result; an empty winner tuple means a draw."""

    reason: TerminalReason
    winners: tuple[PlayerId, ...] = ()
    schema_version: int = TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(set(self.winners)) != len(self.winners):
            raise ValueError("terminal winners cannot contain duplicates")
        canonical = tuple(sorted(self.winners, key=lambda player: player.value))
        if canonical != self.winners:
            raise ValueError("terminal winners must be in canonical player order")

    @property
    def is_draw(self) -> bool:
        """Whether the game ended without a winner."""

        return not self.winners


TerminalState = TerminalResult


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete authoritative state at a resumable engine boundary."""

    supply: SupplyState
    players: tuple[PlayerState, PlayerState]
    normal_achievements: NormalAchievementState
    removed_cards: tuple[CardId, ...]
    phase: GamePhase
    active_player: PlayerId | None
    turn_number: int
    paid_actions_remaining: int
    turn_counters: TurnCounters
    pending_effects: tuple[EffectFrameState, ...]
    effect_variables: tuple[EffectVariable, ...]
    starting_meld_decision_ids: tuple[int, int]
    starting_meld_choices: tuple[CardId | None, CardId | None]
    next_decision_id: int
    next_event_id: int
    next_dogma_action_id: int
    setup: SetupProvenance
    terminal_result: TerminalResult | None = None
    schema_version: int = STATE_SCHEMA_VERSION
    rules_version: str = RULES_VERSION
    information_policy_version: str = INFORMATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if tuple(player.player_id for player in self.players) != tuple(PlayerId):
            raise ValueError("players must be in canonical player order")
        if self.turn_number < 0 or self.paid_actions_remaining < 0:
            raise ValueError("turn fields cannot be negative")
        if len(self.starting_meld_decision_ids) != len(PlayerId):
            raise ValueError("starting meld decision IDs must be in canonical player order")
        if len(set(self.starting_meld_decision_ids)) != len(PlayerId):
            raise ValueError("starting meld decision IDs must be unique")
        if any(decision_id < 1 for decision_id in self.starting_meld_decision_ids):
            raise ValueError("starting meld decision IDs must be positive")
        if len(self.starting_meld_choices) != len(PlayerId):
            raise ValueError("starting meld choices must be in canonical player order")
        if self.phase is not GamePhase.STARTING_MELDS and any(self.starting_meld_choices):
            raise ValueError("starting meld choices must be cleared after setup")
        if self.phase is GamePhase.STARTING_MELDS:
            for player, choice in zip(self.players, self.starting_meld_choices, strict=True):
                if choice is not None and choice not in player.hand:
                    raise ValueError("a starting meld choice must be in that player's hand")
        if min(self.next_decision_id, self.next_event_id, self.next_dogma_action_id) < 1:
            raise ValueError("monotonic IDs must start at one")
        if self.phase is GamePhase.TERMINAL and self.terminal_result is None:
            raise ValueError("terminal phase requires a terminal result")
        if self.phase is not GamePhase.TERMINAL and self.terminal_result is not None:
            raise ValueError("non-terminal state cannot have a terminal result")
        if len(set(self.removed_cards)) != len(self.removed_cards):
            raise ValueError("removed cards cannot be duplicated")
        object.__setattr__(self, "removed_cards", tuple(sorted(self.removed_cards, key=str)))

    def player(self, player_id: PlayerId) -> PlayerState:
        """Return one player's authoritative state."""

        return self.players[tuple(PlayerId).index(player_id)]

    def replace_player(self, replacement: PlayerState) -> GameState:
        """Return the state with one player replaced."""

        players = tuple(
            replacement if player.player_id is replacement.player_id else player
            for player in self.players
        )
        return _replace_game_state(self, players=players)


def _replace_game_state(state: GameState, **changes: object) -> GameState:
    values = {field.name: getattr(state, field.name) for field in fields(GameState)}
    values.update(changes)
    return GameState(**values)


def build_setup_state(seed: int, registry: CardRegistry | None = None) -> GameState:
    """Shuffle supplies, set aside achievements, and deal two cards per player."""

    registry = registry or load_card_registry()
    rng = random.Random(seed)
    shuffled: list[tuple[CardId, ...]] = []
    for age in range(1, 11):
        cards = sorted((card.id for card in registry.cards if card.age == age), key=str)
        rng.shuffle(cards)
        shuffled.append(tuple(cards))
    return build_setup_state_from_piles(tuple(shuffled), seed=seed, registry=registry)


def build_setup_state_from_piles(
    shuffled_piles: tuple[tuple[CardId, ...], ...],
    *,
    seed: int,
    registry: CardRegistry | None = None,
) -> GameState:
    """Build setup from an explicit top-to-bottom order for portable exact replay.

    One card is dealt to each player in canonical player order for two rounds after the age 1-9
    achievements are removed from the tops of their piles.
    """

    registry = registry or load_card_registry()
    if len(shuffled_piles) != 10:
        raise ValueError("explicit setup must contain ten shuffled piles")
    for age, pile in enumerate(shuffled_piles, start=1):
        expected = {card.id for card in registry.cards if card.age == age}
        if len(pile) != len(expected) or set(pile) != expected:
            raise ValueError(f"explicit age {age} pile does not match the card registry")

    working = [list(pile) for pile in shuffled_piles]
    achievement_cards = tuple(working[age - 1].pop(0) for age in range(1, 10))
    hands: dict[PlayerId, list[CardId]] = {player_id: [] for player_id in PlayerId}
    deal_sequence = tuple(player_id for _ in range(2) for player_id in PlayerId)
    for player_id in deal_sequence:
        hands[player_id].append(working[0].pop(0))

    players = tuple(
        PlayerState(player_id, tuple(hands[player_id]), Board.empty(), ()) for player_id in PlayerId
    )
    state = GameState(
        supply=SupplyState(tuple(tuple(pile) for pile in working)),
        players=players,  # type: ignore[arg-type]
        normal_achievements=NormalAchievementState(achievement_cards),
        removed_cards=(),
        phase=GamePhase.STARTING_MELDS,
        active_player=None,
        turn_number=0,
        paid_actions_remaining=0,
        turn_counters=TurnCounters.empty(),
        pending_effects=(),
        effect_variables=(),
        starting_meld_decision_ids=(1, 2),
        starting_meld_choices=(None, None),
        next_decision_id=3,
        next_event_id=1,
        next_dogma_action_id=1,
        setup=SetupProvenance(
            seed,
            registry.data_fingerprint,
            shuffled_piles,
            deal_sequence,
        ),
    )
    from innovation_ai.innovation.zones import assert_state_invariants

    assert_state_invariants(state, registry)
    return state


def clone_state(state: GameState) -> GameState:
    """Return a detached, equal clone suitable for speculative transitions."""

    return copy.deepcopy(state)


def _canonical(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"state contains non-serializable value: {type(value).__name__}")


def state_payload(state: GameState) -> dict[str, object]:
    """Return the canonical JSON-compatible authoritative state payload."""

    payload = _canonical(state)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise TypeError("game state did not serialize to an object")
    return payload


def state_hash(state: GameState) -> str:
    """Return a deterministic SHA-256 hash of the full authoritative state."""

    encoded = json.dumps(
        state_payload(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
