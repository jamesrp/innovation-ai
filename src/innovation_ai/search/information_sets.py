"""Player-safe information-set specifications and deterministic search samples.

Only the trusted builder in this module accepts an authoritative :class:`GameState`.  The
sampler consumes an immutable specification and a private seed; it never has a live-state fallback.
Version 1 intentionally supports only the ``public-covered-v1`` observation policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import cast

from innovation_ai.harness.policy import PublicBoundary, public_boundary
from innovation_ai.innovation.actions import Decision, DecisionKind, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.observations import GameObservation, InformationPolicy, observe
from innovation_ai.innovation.protocol import current_decisions
from innovation_ai.innovation.state import (
    PUBLIC_COVERED_INFORMATION_POLICY_VERSION,
    RULES_VERSION,
    Board,
    ColorStack,
    EffectFrameState,
    EffectVariable,
    GamePhase,
    GameState,
    NormalAchievementState,
    PlayerState,
    PlayerTurnCounters,
    RevealedCard,
    SetupProvenance,
    StateValue,
    SupplyState,
    TurnCounters,
)
from innovation_ai.innovation.types import CardId, NormalAchievementId, PlayerId
from innovation_ai.innovation.zones import (
    StateInvariantError,
    ZoneKind,
    assert_state_invariants,
    locate_card,
)

INFORMATION_SET_SPEC_SCHEMA_VERSION = 1
INFORMATION_SET_SPEC_VERSION = "player-safe-search-spec-v1"
INFORMATION_SET_SAMPLER_VERSION = "player-safe-search-sampler-v1"
INFORMATION_SET_RNG_VERSION = "sha256-counter-v1"
SYNTHETIC_SETUP_RNG_VERSION = "search-information-set-synthetic-setup-v1"
SYNTHETIC_SETUP_SEED = -4


class InformationSetError(RuntimeError):
    """Base class for safe specification and sampling failures."""


class InformationSetSpecError(InformationSetError):
    """The live boundary or an externally supplied specification is malformed."""


class UnsupportedInformationSet(InformationSetSpecError):
    """The boundary cannot be represented by the version-1 safe specification."""


class SamplingError(InformationSetError):
    """Base class for synthetic-state allocation failures."""


class SamplingExhausted(SamplingError):
    """No deterministic allocation attempt reproduced the specification."""


class SampleVerificationError(SamplingError):
    """A synthetic state failed its player-safe contract."""


class _AttemptFailure(SamplingError):
    pass


class HiddenCardDomainKind(StrEnum):
    """Current locations safe to disclose for a hidden runtime card reference."""

    OPPONENT_HAND = "opponent-hand"
    OPPONENT_SCORE = "opponent-score"
    SUPPLY = "supply-age"
    NORMAL_ACHIEVEMENT = "normal-age"
    REMOVED = "removed"


class HiddenCardRole(StrEnum):
    """Structural role at a hidden card token's first occurrence."""

    STARTING_CHOICE = "starting-choice"
    FRAME_SOURCE = "frame-source"
    FRAME_VARIABLE = "frame-variable"
    EFFECT_VARIABLE = "effect-variable"


@dataclass(frozen=True, slots=True)
class HiddenCardDomain:
    """A public-compatible current location, never a hidden card identity."""

    kind: HiddenCardDomainKind
    age: int | None = None

    def __post_init__(self) -> None:
        needs_age = self.kind in {
            HiddenCardDomainKind.SUPPLY,
            HiddenCardDomainKind.NORMAL_ACHIEVEMENT,
        }
        if (self.age is not None) != needs_age:
            raise ValueError(f"{self.kind.value} has an invalid age qualifier")
        if self.age is not None and not 1 <= self.age <= 10:
            raise ValueError("hidden-card domain age must be in 1..10")


@dataclass(frozen=True, slots=True)
class HiddenCardToken:
    """Identity-free description of one aliased hidden card reference."""

    index: int
    role: HiddenCardRole
    domain: HiddenCardDomain

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("hidden-card token index cannot be negative")


@dataclass(frozen=True, slots=True)
class HiddenCardTokenRef:
    """Reference to a token in :attr:`InformationSetSpec.hidden_card_tokens`."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("hidden-card token reference cannot be negative")


# Runtime values retain ordinary strings exactly.  Only strings which are catalog card IDs are
# replaced by token references.
type TokenizedValue = str | int | bool | HiddenCardTokenRef | tuple[TokenizedValue, ...] | None
type TokenizedCard = CardId | HiddenCardTokenRef


@dataclass(frozen=True, slots=True)
class TokenizedEffectVariable:
    """One effect variable whose hidden card strings have been replaced by tokens."""

    name: str
    value: TokenizedValue

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tokenized effect-variable name cannot be empty")


@dataclass(frozen=True, slots=True)
class TokenizedEffectFrame:
    """One resumable effect frame without hidden card identities."""

    kind: str
    step: int
    source_card: TokenizedCard | None
    variables: tuple[TokenizedEffectVariable, ...]

    def __post_init__(self) -> None:
        if not self.kind or self.step < 0:
            raise ValueError("invalid tokenized effect frame")


@dataclass(frozen=True, slots=True)
class PublicScalarContinuation:
    """Non-card authoritative fields needed to resume the exact public boundary."""

    phase: GamePhase
    active_player: PlayerId | None
    turn_number: int
    paid_actions_remaining: int
    starting_meld_decision_ids: tuple[int, int]
    next_decision_id: int
    next_event_id: int
    next_dogma_action_id: int

    def __post_init__(self) -> None:
        if self.phase is GamePhase.TERMINAL:
            raise ValueError("an information set cannot be terminal")
        if self.turn_number < 0 or self.paid_actions_remaining < 0:
            raise ValueError("continuation turn values cannot be negative")
        if len(set(self.starting_meld_decision_ids)) != 2:
            raise ValueError("starting-meld decision IDs must be unique")
        if (
            min(
                *self.starting_meld_decision_ids,
                self.next_decision_id,
                self.next_event_id,
                self.next_dogma_action_id,
            )
            < 1
        ):
            raise ValueError("continuation IDs must be positive")


@dataclass(frozen=True, slots=True)
class TokenizedEffectRuntime:
    """Identity-safe resumable runtime and simultaneous setup commitments."""

    starting_meld_choices: tuple[TokenizedCard | None, TokenizedCard | None]
    pending_effects: tuple[TokenizedEffectFrame, ...]
    effect_variables: tuple[TokenizedEffectVariable, ...]
    revealed: tuple[RevealedCard, ...]


@dataclass(frozen=True, slots=True)
class InformationSetSpec:
    """Immutable complete contract for one player-safe search root."""

    chooser: PlayerId
    observation: GameObservation
    boundary: PublicBoundary
    legal_actions: tuple[SemanticAction, ...]
    continuation: PublicScalarContinuation
    runtime: TokenizedEffectRuntime
    hidden_card_tokens: tuple[HiddenCardToken, ...]
    catalog_fingerprint: str
    rules_version: str
    information_policy_version: str
    target_decision_id: int
    spec_version: str = INFORMATION_SET_SPEC_VERSION
    sampler_version: str = INFORMATION_SET_SAMPLER_VERSION
    rng_version: str = INFORMATION_SET_RNG_VERSION
    schema_version: int = INFORMATION_SET_SPEC_SCHEMA_VERSION
    spec_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != INFORMATION_SET_SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported information-set schema")
        if self.spec_version != INFORMATION_SET_SPEC_VERSION:
            raise ValueError("unsupported information-set specification version")
        if self.sampler_version != INFORMATION_SET_SAMPLER_VERSION:
            raise ValueError("unsupported information-set sampler version")
        if self.rng_version != INFORMATION_SET_RNG_VERSION:
            raise ValueError("unsupported information-set RNG version")
        if self.observation.viewer is not self.chooser:
            raise ValueError("information-set chooser does not own its observation")
        if self.information_policy_version != PUBLIC_COVERED_INFORMATION_POLICY_VERSION:
            raise ValueError("search supports only public-covered-v1")
        if self.observation.information_policy is not InformationPolicy.PUBLIC_COVERED:
            raise ValueError("information-set observation is not public-covered-v1")
        if self.observation.rules_version != self.rules_version:
            raise ValueError("observation rules version differs from specification")
        if self.target_decision_id < 1 or not self.legal_actions:
            raise ValueError("information set requires one non-empty target decision")
        if any(action.decision_id != self.target_decision_id for action in self.legal_actions):
            raise ValueError("legal actions differ from target decision ID")
        visible = _visible_card_ids(self.observation)
        if any(
            card_id not in visible
            for card_id in _card_ids_in_contract((self.boundary, self.legal_actions))
        ):
            raise ValueError("information-set public contracts contain a hidden card identity")
        if tuple(token.index for token in self.hidden_card_tokens) != tuple(
            range(len(self.hidden_card_tokens))
        ):
            raise ValueError("hidden-card tokens must be in first-occurrence order")
        _validate_token_references(self.runtime, len(self.hidden_card_tokens))
        digest = _tagged_digest(self)
        object.__setattr__(self, "spec_digest", digest)

    @property
    def digest(self) -> str:
        """Return the content-derived specification digest."""

        return self.spec_digest


class _Sha256Rng:
    def __init__(self, seed: int | str | bytes, *domain: str) -> None:
        self._seed = _seed_bytes(seed)
        self._domain = b"\0".join(item.encode("ascii") for item in domain)
        self._counter = 0
        self._buffer = b""

    def _bytes(self, count: int) -> bytes:
        while len(self._buffer) < count:
            payload = (
                b"innovation-ai/"
                + INFORMATION_SET_RNG_VERSION.encode("ascii")
                + b"\0"
                + self._seed
                + b"\0"
                + self._domain
                + b"\0"
                + self._counter.to_bytes(16, "big")
            )
            self._buffer += hashlib.sha256(payload).digest()
            self._counter += 1
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def randbelow(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("random range must be positive")
        width = max(1, ((stop - 1).bit_length() + 7) // 8)
        ceiling = 1 << (width * 8)
        accepted = ceiling - ceiling % stop
        while True:
            value = int.from_bytes(self._bytes(width), "big")
            if value < accepted:
                return value % stop

    def shuffled(self, values: Iterable[CardId]) -> list[CardId]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            other = self.randbelow(index + 1)
            result[index], result[other] = result[other], result[index]
        return result


def _seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bool):
        raise TypeError("information-set seed cannot be a boolean")
    if isinstance(seed, int):
        return f"int:{seed}".encode("ascii")
    if isinstance(seed, str):
        return b"str:" + seed.encode()
    if isinstance(seed, bytes):
        return b"bytes:" + seed
    raise TypeError("information-set seed must be int, str, or bytes")


def _canonical(value: object) -> object:
    if isinstance(value, CardId):
        return {"$card": value.value}
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
            if item.name != "spec_digest"
        }
        result["$type"] = type(value).__name__
        return result
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"information-set value is not canonicalizable: {type(value).__name__}")


def _tagged_digest(spec: InformationSetSpec) -> str:
    payload = json.dumps(_canonical(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (
        "sha256:"
        + hashlib.sha256(b"player-safe-search-spec-v1\0" + payload.encode("ascii")).hexdigest()
    )


def _visible_card_ids(observation: GameObservation) -> frozenset[CardId]:
    visible = set(observation.revealed_cards)
    for player in observation.players:
        visible.update(player.hand.known_cards)
        visible.update(player.score_pile.known_cards)
        for stack in player.board:
            if stack.top_card_id is not None:
                visible.add(stack.top_card_id)
            visible.update(
                covered.card_id for covered in stack.covered_cards if covered.card_id is not None
            )
    return frozenset(visible)


def _card_ids_in_contract(value: object) -> Iterable[CardId]:
    if isinstance(value, CardId):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _card_ids_in_contract(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _card_ids_in_contract(item)


class _RuntimeTokenizer:
    def __init__(self, state: GameState, chooser: PlayerId, registry: CardRegistry) -> None:
        self.state = state
        self.chooser = chooser
        self.registry = registry
        self.visible = _visible_card_ids(observe(state, chooser, registry))
        self.by_card: dict[CardId, HiddenCardTokenRef] = {}
        self.tokens: list[HiddenCardToken] = []

    def card(self, card_id: CardId, role: HiddenCardRole) -> TokenizedCard:
        if card_id in self.visible:
            return card_id
        existing = self.by_card.get(card_id)
        if existing is not None:
            return existing
        domain = self._domain(card_id)
        reference = HiddenCardTokenRef(len(self.tokens))
        self.by_card[card_id] = reference
        self.tokens.append(HiddenCardToken(reference.index, role, domain))
        return reference

    def value(self, value: StateValue, role: HiddenCardRole) -> TokenizedValue:
        if isinstance(value, tuple):
            return tuple(self.value(item, role) for item in value)
        if isinstance(value, str):
            try:
                card_id = CardId(value)
            except ValueError:
                return value
            if card_id in self.registry.by_id:
                tokenized = self.card(card_id, role)
                return tokenized if isinstance(tokenized, HiddenCardTokenRef) else tokenized.value
        return value

    def variable(self, variable: EffectVariable, role: HiddenCardRole) -> TokenizedEffectVariable:
        return TokenizedEffectVariable(variable.name, self.value(variable.value, role))

    def _domain(self, card_id: CardId) -> HiddenCardDomain:
        location = locate_card(self.state, card_id)
        opponent = _other_player(self.chooser)
        if location.kind is ZoneKind.HAND and location.player_id is opponent:
            return HiddenCardDomain(HiddenCardDomainKind.OPPONENT_HAND)
        if location.kind is ZoneKind.SCORE and location.player_id is opponent:
            return HiddenCardDomain(HiddenCardDomainKind.OPPONENT_SCORE)
        if location.kind is ZoneKind.SUPPLY and location.age is not None:
            return HiddenCardDomain(HiddenCardDomainKind.SUPPLY, location.age)
        if location.kind is ZoneKind.NORMAL_ACHIEVEMENT:
            assert location.normal_achievement_id is not None
            age = tuple(NormalAchievementId).index(location.normal_achievement_id) + 1
            return HiddenCardDomain(HiddenCardDomainKind.NORMAL_ACHIEVEMENT, age)
        if location.kind is ZoneKind.REMOVED:
            return HiddenCardDomain(HiddenCardDomainKind.REMOVED)
        raise UnsupportedInformationSet(
            f"hidden runtime card {card_id} has no player-safe token domain"
        )


class InformationSetSpecBuilder:
    """Trusted bridge from an authoritative state and its exact decision."""

    def __init__(
        self,
        registry: CardRegistry | None = None,
        programs: EffectProgramRegistry | None = None,
    ) -> None:
        self._registry = registry or load_card_registry()
        self._programs = programs or load_effect_programs()

    def build(self, state: GameState, decision: Decision) -> InformationSetSpec:
        registry = self._registry
        if state.information_policy_version != PUBLIC_COVERED_INFORMATION_POLICY_VERSION:
            raise UnsupportedInformationSet("search supports only public-covered-v1")
        if state.phase is GamePhase.TERMINAL or state.terminal_result is not None:
            raise UnsupportedInformationSet("terminal states have no searchable information set")
        try:
            assert_state_invariants(state, registry)
        except StateInvariantError as error:
            raise InformationSetSpecError(f"live state violates invariants: {error}") from error
        candidates = current_decisions(state, registry, self._programs)
        exact = next(
            (item for item in candidates if item.decision_id == decision.decision_id), None
        )
        if exact != decision:
            raise InformationSetSpecError("supplied decision is not the exact live decision")
        if decision.kind not in {
            DecisionKind.TURN_ACTION,
            DecisionKind.STARTING_MELD,
            DecisionKind.EFFECT_CHOICE,
        }:
            raise UnsupportedInformationSet(f"unsupported decision kind {decision.kind}")
        observation = observe(state, decision.chooser, registry)
        if observation != decision.observation:
            raise InformationSetSpecError("decision observation is stale")
        _validate_observation(observation, registry)
        visible = _visible_card_ids(observation)
        hidden_public_metadata = tuple(
            card_id
            for card_id in _card_ids_in_contract(
                (
                    decision.legal_actions,
                    public_boundary(state, decision.chooser, decision, observation),
                )
            )
            if card_id not in visible
        )
        if hidden_public_metadata:
            raise InformationSetSpecError(
                "decision public metadata contains card identities hidden from chooser"
            )

        tokenizer = _RuntimeTokenizer(state, decision.chooser, registry)
        choices = tuple(
            None if card_id is None else tokenizer.card(card_id, HiddenCardRole.STARTING_CHOICE)
            for card_id in state.starting_meld_choices
        )
        frames = tuple(
            TokenizedEffectFrame(
                frame.kind,
                frame.step,
                None
                if frame.source_card_id is None
                else tokenizer.card(frame.source_card_id, HiddenCardRole.FRAME_SOURCE),
                tuple(
                    tokenizer.variable(variable, HiddenCardRole.FRAME_VARIABLE)
                    for variable in frame.variables
                ),
            )
            for frame in state.pending_effects
        )
        variables = tuple(
            tokenizer.variable(variable, HiddenCardRole.EFFECT_VARIABLE)
            for variable in state.effect_variables
        )
        # A reveal is visible by definition.  Keeping its public scope is necessary to resume the
        # real effect VM and does not disclose a hidden card identity.
        if any(item.card_id not in visible for item in state.revealed):
            raise InformationSetSpecError("effect runtime contains a non-public reveal marker")
        continuation = PublicScalarContinuation(
            state.phase,
            state.active_player,
            state.turn_number,
            state.paid_actions_remaining,
            state.starting_meld_decision_ids,
            state.next_decision_id,
            state.next_event_id,
            state.next_dogma_action_id,
        )
        return InformationSetSpec(
            chooser=decision.chooser,
            observation=observation,
            boundary=public_boundary(state, decision.chooser, decision, observation),
            legal_actions=decision.legal_actions,
            continuation=continuation,
            runtime=TokenizedEffectRuntime(
                cast(tuple[TokenizedCard | None, TokenizedCard | None], choices),
                frames,
                variables,
                state.revealed,
            ),
            hidden_card_tokens=tuple(tokenizer.tokens),
            catalog_fingerprint=registry.data_fingerprint,
            rules_version=state.rules_version,
            information_policy_version=state.information_policy_version,
            target_decision_id=decision.decision_id,
        )


@dataclass(slots=True)
class _AllocatedPosition:
    players: tuple[PlayerState, PlayerState]
    supply: SupplyState
    normal: NormalAchievementState
    removed: tuple[CardId, ...]


def _other_player(player_id: PlayerId) -> PlayerId:
    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def _validate_observation(observation: GameObservation, registry: CardRegistry) -> None:
    if observation.information_policy is not InformationPolicy.PUBLIC_COVERED:
        raise InformationSetSpecError("search observation must use public-covered-v1")
    if tuple(item.age for item in observation.supplies) != tuple(range(1, 11)):
        raise InformationSetSpecError("supplies are not in canonical age order")
    if tuple(player.player_id for player in observation.players) != tuple(PlayerId):
        raise InformationSetSpecError("players are not in canonical order")
    seen: set[CardId] = set()
    for player in observation.players:
        for zone in (player.hand, player.score_pile):
            if tuple(sorted(zone.values)) != zone.values or any(
                not 1 <= age <= 10 for age in zone.values
            ):
                raise InformationSetSpecError("zone age multiset is malformed")
            for card_id in zone.known_cards:
                if card_id in seen or registry.card(card_id).age not in zone.values:
                    raise InformationSetSpecError("known zone card is duplicated or has wrong age")
                seen.add(card_id)
        for stack in player.board:
            if stack.top_card_id is None:
                if stack.covered_cards or stack.covered_count not in (None, 0):
                    raise InformationSetSpecError("empty board stack has covered cards")
                continue
            ordered = (*stack.covered_cards,)
            if stack.covered_count != len(ordered):
                raise InformationSetSpecError("public-covered stack count is not exact")
            card_ids = tuple(card.card_id for card in ordered)
            if any(card_id is None for card_id in card_ids):
                raise InformationSetSpecError("public-covered board contains a hidden identity")
            for card_id in (*cast(tuple[CardId, ...], card_ids), stack.top_card_id):
                card = registry.card(card_id)
                if card.color is not stack.color or card_id in seen:
                    raise InformationSetSpecError("board identity is duplicated or in wrong color")
                seen.add(card_id)
    viewer = observation.player(observation.viewer)
    if (
        len(viewer.hand.known_cards) != viewer.hand.count
        or len(viewer.score_pile.known_cards) != viewer.score_pile.count
    ):
        raise InformationSetSpecError("chooser hand and score identities must be exact")


def _unknown_zone_ages(
    values: tuple[int, ...], known: tuple[CardId, ...], registry: CardRegistry
) -> tuple[int, ...]:
    public = Counter(values)
    exact = Counter(registry.card(card_id).age for card_id in known)
    if exact - public:
        raise InformationSetSpecError("known cards do not fit a zone age multiset")
    return tuple(sorted((public - exact).elements()))


def _fixed_cards(observation: GameObservation) -> set[CardId]:
    result: set[CardId] = set()
    for player in observation.players:
        result.update(player.hand.known_cards)
        result.update(player.score_pile.known_cards)
        for stack in player.board:
            if stack.top_card_id is not None:
                result.add(stack.top_card_id)
            result.update(cast(CardId, card.card_id) for card in stack.covered_cards)
    return result


def _allocate_position(
    spec: InformationSetSpec, registry: CardRegistry, rng: _Sha256Rng
) -> _AllocatedPosition:
    observation = spec.observation
    fixed = _fixed_cards(observation)
    if len(fixed) != sum(
        len(player.hand.known_cards)
        + len(player.score_pile.known_cards)
        + sum(
            (1 if stack.top_card_id is not None else 0) + len(stack.covered_cards)
            for stack in player.board
        )
        for player in observation.players
    ):
        raise InformationSetSpecError("a visible card occurs in multiple observed locations")
    available_by_age: dict[int, list[CardId]] = {}
    for age in range(1, 11):
        available_by_age[age] = rng.shuffled(
            sorted(
                (card.id for card in registry.cards if card.age == age and card.id not in fixed),
                key=str,
            )
        )

    allocated: defaultdict[str, list[CardId]] = defaultdict(list)

    def take(age: int, key: str, count: int) -> None:
        if count < 0 or len(available_by_age[age]) < count:
            raise _AttemptFailure(f"not enough age-{age} cards for {key}")
        allocated[key].extend(available_by_age[age][:count])
        del available_by_age[age][:count]

    for age in range(1, 10):
        take(age, f"normal:{age}", 1)
    for supply in observation.supplies:
        take(supply.age, f"supply:{supply.age}", supply.count)
    opponent = _other_player(spec.chooser)
    opponent_observation = observation.player(opponent)
    for age in _unknown_zone_ages(
        opponent_observation.hand.values, opponent_observation.hand.known_cards, registry
    ):
        take(age, "opponent-hand", 1)
    for age in _unknown_zone_ages(
        opponent_observation.score_pile.values,
        opponent_observation.score_pile.known_cards,
        registry,
    ):
        take(age, "opponent-score", 1)

    players: list[PlayerState] = []
    for observed in observation.players:
        hand = list(observed.hand.known_cards)
        score = list(observed.score_pile.known_cards)
        if observed.player_id is opponent:
            hand.extend(allocated["opponent-hand"])
            score.extend(allocated["opponent-score"])
        stacks = []
        for stack in observed.board:
            covered = tuple(cast(CardId, item.card_id) for item in stack.covered_cards)
            cards = covered + ((stack.top_card_id,) if stack.top_card_id is not None else ())
            stacks.append(ColorStack(stack.color, cards, stack.splay))
        players.append(
            PlayerState(
                observed.player_id,
                tuple(hand),
                Board(tuple(stacks)),
                tuple(score),
                observed.normal_achievements,
                observed.special_achievements,
            )
        )
    piles = tuple(tuple(rng.shuffled(allocated[f"supply:{age}"])) for age in range(1, 11))
    normal = tuple(allocated[f"normal:{age}"][0] for age in range(1, 10))
    removed = tuple(
        sorted((card for cards in available_by_age.values() for card in cards), key=str)
    )
    return _AllocatedPosition(
        cast(tuple[PlayerState, PlayerState], tuple(players)),
        SupplyState(piles),
        NormalAchievementState(normal),
        removed,
    )


def _domain_cards(
    domain: HiddenCardDomain, position: _AllocatedPosition, spec: InformationSetSpec
) -> tuple[CardId, ...]:
    opponent_id = _other_player(spec.chooser)
    opponent = position.players[tuple(PlayerId).index(opponent_id)]
    observed_opponent = spec.observation.player(opponent_id)
    if domain.kind is HiddenCardDomainKind.OPPONENT_HAND:
        return tuple(
            card for card in opponent.hand if card not in observed_opponent.hand.known_cards
        )
    if domain.kind is HiddenCardDomainKind.OPPONENT_SCORE:
        return tuple(
            card
            for card in opponent.score_pile
            if card not in observed_opponent.score_pile.known_cards
        )
    if domain.kind is HiddenCardDomainKind.SUPPLY:
        assert domain.age is not None
        return position.supply.pile(domain.age)
    if domain.kind is HiddenCardDomainKind.NORMAL_ACHIEVEMENT:
        assert domain.age is not None
        return (position.normal.cards[domain.age - 1],)
    return position.removed


def _bind_tokens(
    spec: InformationSetSpec, position: _AllocatedPosition, rng: _Sha256Rng
) -> dict[int, CardId]:
    bindings: dict[int, CardId] = {}
    used: set[CardId] = set()
    shuffled_domains: dict[HiddenCardDomain, list[CardId]] = {}
    for token in spec.hidden_card_tokens:
        candidates = shuffled_domains.setdefault(
            token.domain,
            rng.shuffled(sorted(_domain_cards(token.domain, position, spec), key=str)),
        )
        selected = next((card for card in candidates if card not in used), None)
        if selected is None:
            raise _AttemptFailure(f"token domain {token.domain.kind.value} is exhausted")
        bindings[token.index] = selected
        used.add(selected)
    return bindings


def _materialize_card(value: TokenizedCard, bindings: dict[int, CardId]) -> CardId:
    return bindings[value.index] if isinstance(value, HiddenCardTokenRef) else value


def _materialize_value(value: TokenizedValue, bindings: dict[int, CardId]) -> StateValue:
    if isinstance(value, HiddenCardTokenRef):
        return bindings[value.index].value
    if isinstance(value, tuple):
        return tuple(_materialize_value(item, bindings) for item in value)
    return value


def _materialize_runtime(
    runtime: TokenizedEffectRuntime, bindings: dict[int, CardId]
) -> tuple[
    tuple[CardId | None, CardId | None],
    tuple[EffectFrameState, ...],
    tuple[EffectVariable, ...],
]:
    choices = tuple(
        None if item is None else _materialize_card(item, bindings)
        for item in runtime.starting_meld_choices
    )
    frames = tuple(
        EffectFrameState(
            item.kind,
            item.step,
            None if item.source_card is None else _materialize_card(item.source_card, bindings),
            tuple(
                EffectVariable(variable.name, _materialize_value(variable.value, bindings))
                for variable in item.variables
            ),
        )
        for item in runtime.pending_effects
    )
    variables = tuple(
        EffectVariable(item.name, _materialize_value(item.value, bindings))
        for item in runtime.effect_variables
    )
    return cast(tuple[CardId | None, CardId | None], choices), frames, variables


def _synthetic_setup(registry: CardRegistry) -> SetupProvenance:
    return SetupProvenance(
        SYNTHETIC_SETUP_SEED,
        registry.data_fingerprint,
        tuple(
            tuple(sorted((card.id for card in registry.cards if card.age == age), key=str))
            for age in range(1, 11)
        ),
        tuple(player for _ in range(2) for player in PlayerId),
        rng_version=SYNTHETIC_SETUP_RNG_VERSION,
    )


def _validate_token_references(runtime: TokenizedEffectRuntime, count: int) -> None:
    references: list[int] = []

    def walk(value: object) -> None:
        if isinstance(value, HiddenCardTokenRef):
            references.append(value.index)
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                walk(getattr(value, item.name))
        elif isinstance(value, tuple):
            for item in value:
                walk(item)

    walk(runtime)
    if any(index >= count for index in references):
        raise ValueError("runtime references an unknown hidden-card token")
    if set(references) != set(range(count)):
        raise ValueError("every hidden-card token must be referenced")


class InformationSetSampler:
    """Deterministically materialize synthetic states from a spec and seed only."""

    def __init__(
        self,
        registry: CardRegistry | None = None,
        programs: EffectProgramRegistry | None = None,
        *,
        seed: int | str | bytes | None = None,
        retry_limit: int = 32,
        strict: bool = True,
    ) -> None:
        if retry_limit < 1:
            raise ValueError("retry limit must be positive")
        if seed is not None:
            _seed_bytes(seed)
        self._registry = registry or load_card_registry()
        self._programs = programs or load_effect_programs()
        self._default_seed = seed
        self._retry_limit = retry_limit
        self._strict = strict

    def sample(
        self,
        spec: InformationSetSpec,
        seed: int | str | bytes | None = None,
        *,
        sample_index: int = 0,
    ) -> GameState | None:
        if sample_index < 0:
            raise ValueError("sample index cannot be negative")
        resolved_seed = self._resolved_seed(seed)
        self._validate_spec(spec)
        failures: list[str] = []
        for attempt in range(self._retry_limit):
            rng = _Sha256Rng(
                resolved_seed, "search-sample", spec.spec_digest, str(sample_index), str(attempt)
            )
            try:
                position = _allocate_position(spec, self._registry, rng)
                bindings = _bind_tokens(spec, position, rng)
                choices, frames, variables = _materialize_runtime(spec.runtime, bindings)
                continuation = spec.continuation
                progress = spec.boundary.turn_progress
                counters = TurnCounters(
                    cast(
                        tuple[PlayerTurnCounters, PlayerTurnCounters],
                        tuple(
                            PlayerTurnCounters(
                                player,
                                tucked=(
                                    progress.self_tucked
                                    if player is spec.chooser
                                    else progress.opponent_tucked
                                ),
                                scored=(
                                    progress.self_scored
                                    if player is spec.chooser
                                    else progress.opponent_scored
                                ),
                            )
                            for player in PlayerId
                        ),
                    )
                )
                state = GameState(
                    supply=position.supply,
                    players=position.players,
                    normal_achievements=position.normal,
                    removed_cards=position.removed,
                    phase=continuation.phase,
                    active_player=continuation.active_player,
                    turn_number=continuation.turn_number,
                    paid_actions_remaining=continuation.paid_actions_remaining,
                    turn_counters=counters,
                    pending_effects=frames,
                    effect_variables=variables,
                    revealed=spec.runtime.revealed,
                    starting_meld_decision_ids=continuation.starting_meld_decision_ids,
                    starting_meld_choices=choices,
                    next_decision_id=continuation.next_decision_id,
                    next_event_id=continuation.next_event_id,
                    next_dogma_action_id=continuation.next_dogma_action_id,
                    setup=_synthetic_setup(self._registry),
                    rules_version=spec.rules_version,
                    information_policy_version=spec.information_policy_version,
                )
                verify_sampled_state(spec, state, self._registry, self._programs)
                return state
            except (_AttemptFailure, SampleVerificationError, ValueError) as error:
                failures.append(str(error))
        exhausted = SamplingExhausted(
            f"sample {sample_index} exhausted {self._retry_limit} deterministic attempts: "
            f"{failures[-1] if failures else 'unknown allocation failure'}"
        )
        if self._strict:
            raise exhausted
        return None

    def sample_many(
        self,
        spec: InformationSetSpec,
        count: int,
        seed: int | str | bytes | None = None,
    ) -> tuple[GameState | None, ...]:
        if count < 0:
            raise ValueError("sample count cannot be negative")
        resolved_seed = self._resolved_seed(seed)
        return tuple(self.sample(spec, resolved_seed, sample_index=index) for index in range(count))

    def _resolved_seed(self, seed: int | str | bytes | None) -> int | str | bytes:
        result = self._default_seed if seed is None else seed
        if result is None:
            raise TypeError("sample requires an explicit seed")
        _seed_bytes(result)
        return result

    def _validate_spec(self, spec: InformationSetSpec) -> None:
        if spec.spec_digest != _tagged_digest(spec):
            raise InformationSetSpecError("information-set digest is invalid")
        if spec.catalog_fingerprint != self._registry.data_fingerprint:
            raise InformationSetSpecError("specification catalog differs from real registry")
        if spec.rules_version != RULES_VERSION:
            raise InformationSetSpecError("specification rules version is unsupported")
        if spec.information_policy_version != PUBLIC_COVERED_INFORMATION_POLICY_VERSION:
            raise InformationSetSpecError("search supports only public-covered-v1")
        _validate_observation(spec.observation, self._registry)
        visible = _visible_card_ids(spec.observation)
        if any(card_id not in visible for card_id in _card_ids_in_contract(spec.legal_actions)):
            raise InformationSetSpecError("specification legal actions disclose a hidden card")


def verify_sampled_state(
    spec: InformationSetSpec,
    sampled: GameState,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> None:
    """Verify conservation, observation, target decision, and public boundary exactly."""

    registry = registry or load_card_registry()
    programs = programs or load_effect_programs()
    if sampled.setup != _synthetic_setup(registry):
        raise SampleVerificationError("sample does not have synthetic setup provenance")
    try:
        assert_state_invariants(sampled, registry)
    except StateInvariantError as error:
        raise SampleVerificationError(f"sample violates state invariants: {error}") from error
    observation = observe(sampled, spec.chooser, registry)
    if observation != spec.observation:
        raise SampleVerificationError("sample does not reproduce the chooser observation")
    decisions = current_decisions(sampled, registry, programs)
    decision = next(
        (item for item in decisions if item.decision_id == spec.target_decision_id), None
    )
    if decision is None:
        raise SampleVerificationError("sample does not reproduce the target decision ID")
    if decision.kind is not spec.boundary.decision_kind:
        raise SampleVerificationError("sample decision kind differs from target")
    if decision.legal_actions != spec.legal_actions:
        raise SampleVerificationError("sample does not reproduce exact legal actions")
    boundary = public_boundary(sampled, spec.chooser, decision, observation)
    if boundary != spec.boundary:
        raise SampleVerificationError("sample does not reproduce exact public decision metadata")


def build_information_set_spec(
    state: GameState,
    decision: Decision,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> InformationSetSpec:
    """Functional trusted-builder entry point."""

    return InformationSetSpecBuilder(registry, programs).build(state, decision)


def sample_information_set(
    spec: InformationSetSpec,
    seed: int | str | bytes,
    *,
    sample_index: int = 0,
    retry_limit: int = 32,
    strict: bool = True,
) -> GameState | None:
    """Functional sampler whose policy input is exactly ``spec`` plus ``seed``."""

    return InformationSetSampler(retry_limit=retry_limit, strict=strict).sample(
        spec, seed, sample_index=sample_index
    )


__all__ = [
    "INFORMATION_SET_RNG_VERSION",
    "INFORMATION_SET_SAMPLER_VERSION",
    "INFORMATION_SET_SPEC_SCHEMA_VERSION",
    "INFORMATION_SET_SPEC_VERSION",
    "SYNTHETIC_SETUP_RNG_VERSION",
    "SYNTHETIC_SETUP_SEED",
    "HiddenCardDomain",
    "HiddenCardDomainKind",
    "HiddenCardRole",
    "HiddenCardToken",
    "HiddenCardTokenRef",
    "InformationSetError",
    "InformationSetSampler",
    "InformationSetSpec",
    "InformationSetSpecBuilder",
    "InformationSetSpecError",
    "PublicScalarContinuation",
    "SampleVerificationError",
    "SamplingError",
    "SamplingExhausted",
    "TokenizedEffectFrame",
    "TokenizedEffectRuntime",
    "TokenizedEffectVariable",
    "UnsupportedInformationSet",
    "build_information_set_spec",
    "sample_information_set",
    "verify_sampled_state",
]
