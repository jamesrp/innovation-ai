"""Current-information determinization for safe paid-turn afterstates.

The builder is the sole trusted bridge from a live :class:`GameState` to an immutable,
player-safe information-set specification.  Sampling deliberately accepts that specification,
a card registry, and a deterministic private seed only; it never receives an authoritative live
state from which it could copy hidden card identities or supply order.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum

from innovation_ai.harness.policy import (
    PUBLIC_BOUNDARY_SCHEMA_VERSION,
    PublicBoundary,
    public_boundary,
)
from innovation_ai.innovation.actions import (
    AchieveAction,
    DecisionKind,
    DogmaAction,
    DrawAction,
    MeldAction,
    SemanticAction,
)
from innovation_ai.innovation.board import covered_visible_slots
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import (
    CoveredCardObservation,
    GameObservation,
    StackObservation,
    ZoneObservation,
    observe,
)
from innovation_ai.innovation.protocol import current_decision
from innovation_ai.innovation.state import (
    RULES_VERSION,
    SUPPORTED_INFORMATION_POLICY_VERSIONS,
    Board,
    ColorStack,
    GamePhase,
    GameState,
    NormalAchievementState,
    PlayerState,
    PlayerTurnCounters,
    SetupProvenance,
    SupplyState,
    TurnCounters,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    Icon,
    NormalAchievementId,
    PlayerId,
    SplayDirection,
)
from innovation_ai.innovation.zones import StateInvariantError, assert_state_invariants

INFORMATION_SET_SPEC_SCHEMA_VERSION = 1
INFORMATION_SET_SAMPLER_VERSION = "information-set-sampler-v1"
INFORMATION_SET_RNG_VERSION = "sha256-counter-v1"
SYNTHETIC_SETUP_RNG_VERSION = "information-set-synthetic-setup-v1"
SYNTHETIC_SETUP_SEED = -2


class InformationSetError(RuntimeError):
    """Base class for trusted information-set construction and sampling failures."""


class InformationSetSpecError(InformationSetError):
    """A live state or externally supplied information-set spec is unsupported or malformed."""


class UnsupportedInformationSet(InformationSetSpecError):
    """The requested state is not a stable paid-turn information-set boundary."""


class SamplingError(InformationSetError):
    """Base class for failures while reconstructing a sampled authoritative state."""


class SamplingExhausted(SamplingError):
    """All deterministic allocation attempts were infeasible.

    This is intentionally not a signal to inspect or evaluate the original state.  A caller may
    use a separately configured non-clairvoyant fallback, or fail loudly in strict mode.
    """


class SampleVerificationError(SamplingError):
    """A generated state did not reproduce its information-set contract."""


class _SamplingAttemptFailure(SamplingError):
    """One randomized allocation branch could not satisfy its public constraints."""


class HiddenAllocationKind(StrEnum):
    """Publicly describable classes of card identities hidden from the chooser."""

    NORMAL_ACHIEVEMENT = "normal-achievement"
    SUPPLY = "supply"
    OPPONENT_HAND = "opponent-hand"
    OPPONENT_SCORE = "opponent-score"
    OPPONENT_SPLAYED_COVERED = "opponent-splayed-covered"
    OPPONENT_UNSPLAYED_COVERED = "opponent-unsplayed-covered"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class PublicMonotonicIds:
    """The public/audited protocol IDs needed to rebuild a legal turn boundary."""

    starting_meld_decision_ids: tuple[int, int]
    next_decision_id: int
    next_event_id: int
    next_dogma_action_id: int

    def __post_init__(self) -> None:
        if len(set(self.starting_meld_decision_ids)) != len(PlayerId):
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
            raise ValueError("monotonic IDs must be positive")
        if max(self.starting_meld_decision_ids) >= self.next_decision_id:
            raise ValueError("next decision ID must follow starting-meld IDs")


@dataclass(frozen=True, slots=True)
class HiddenAllocationConstraint:
    """One explicit public constraint on hidden card allocation.

    ``count=None`` means an intentionally unobserved, variable-size location.  In particular,
    an opponent's unsplayed covered cards cannot reveal their count under the default information
    policy, so the sampler chooses that count rather than copying it from a real state.
    """

    kind: HiddenAllocationKind
    count: int | None
    age: int | None = None
    color: Color | None = None
    visible_icons: tuple[Icon, ...] = ()
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 0:
            raise ValueError("hidden allocation count cannot be negative")
        if self.age is not None and not 1 <= self.age <= 10:
            raise ValueError("hidden allocation age must be in 1..10")
        if self.ordinal is not None and self.ordinal < 0:
            raise ValueError("hidden allocation ordinal cannot be negative")
        if (
            self.kind
            in {
                HiddenAllocationKind.NORMAL_ACHIEVEMENT,
                HiddenAllocationKind.SUPPLY,
                HiddenAllocationKind.OPPONENT_HAND,
                HiddenAllocationKind.OPPONENT_SCORE,
            }
            and self.age is None
        ):
            raise ValueError(f"{self.kind} constraint requires an age")
        if (
            self.kind
            in {
                HiddenAllocationKind.OPPONENT_SPLAYED_COVERED,
                HiddenAllocationKind.OPPONENT_UNSPLAYED_COVERED,
            }
            and self.color is None
        ):
            raise ValueError(f"{self.kind} constraint requires a color")
        if self.kind is HiddenAllocationKind.OPPONENT_SPLAYED_COVERED and (
            self.count != 1 or self.ordinal is None
        ):
            raise ValueError("a hidden splayed covered-card slot must have count one and ordinal")
        if (
            self.kind
            in {
                HiddenAllocationKind.OPPONENT_UNSPLAYED_COVERED,
                HiddenAllocationKind.REMOVED,
            }
            and self.count is not None
        ):
            raise ValueError(f"{self.kind} count must remain intentionally unknown")


@dataclass(frozen=True, slots=True)
class InformationSetSpec:
    """Immutable, player-safe skeleton from which a sampled state can be reconstructed."""

    chooser: PlayerId
    observation: GameObservation
    boundary: PublicBoundary
    legal_actions: tuple[SemanticAction, ...]
    catalog_fingerprint: str
    rules_version: str
    information_policy_version: str
    monotonic_ids: PublicMonotonicIds
    hidden_allocation_constraints: tuple[HiddenAllocationConstraint, ...]
    sampler_version: str = INFORMATION_SET_SAMPLER_VERSION
    rng_version: str = INFORMATION_SET_RNG_VERSION
    schema_version: int = INFORMATION_SET_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INFORMATION_SET_SPEC_SCHEMA_VERSION:
            raise ValueError(f"unsupported information-set spec schema {self.schema_version}")
        if self.sampler_version != INFORMATION_SET_SAMPLER_VERSION:
            raise ValueError(f"unsupported information-set sampler {self.sampler_version}")
        if self.rng_version != INFORMATION_SET_RNG_VERSION:
            raise ValueError(f"unsupported information-set RNG {self.rng_version}")
        if not self.catalog_fingerprint.startswith("sha256:"):
            raise ValueError("information-set catalog fingerprint must be a SHA-256 digest")
        if self.observation.viewer is not self.chooser:
            raise ValueError("information-set chooser must own its observation")
        if self.observation.phase is not GamePhase.PLAY:
            raise ValueError("information-set observation must be a play boundary")
        if self.observation.active_player is not self.chooser:
            raise ValueError("information-set chooser must be active")
        if self.observation.rules_version != self.rules_version:
            raise ValueError("information-set rules version differs from its observation")
        if self.observation.information_policy.value != self.information_policy_version:
            raise ValueError("information-set policy version differs from its observation")
        if self.boundary.schema_version != PUBLIC_BOUNDARY_SCHEMA_VERSION:
            raise ValueError("information-set public boundary schema is unsupported")
        if self.boundary.decision_kind is not DecisionKind.TURN_ACTION:
            raise ValueError("information-set boundary must describe a turn action")
        if self.boundary.chooser_relation.value != "self":
            raise ValueError("information-set turn chooser must be the viewer")
        if not isinstance(self.legal_actions, tuple):
            raise ValueError("information-set legal actions must be immutable tuples")
        if not self.legal_actions:
            raise ValueError("information-set must contain legal turn actions")
        if any(
            action.decision_id != self.monotonic_ids.next_decision_id
            for action in self.legal_actions
        ):
            raise ValueError("information-set actions do not match the next decision ID")
        if any(
            not isinstance(action, (DrawAction, MeldAction, DogmaAction, AchieveAction))
            for action in self.legal_actions
        ):
            raise ValueError("information-set contains a non-turn action")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("information-set legal actions cannot repeat")
        if tuple(self.hidden_allocation_constraints) != self.hidden_allocation_constraints:
            raise ValueError("information-set constraints must be immutable tuples")

    @property
    def digest(self) -> str:
        """Return a deterministic digest used for domain-separated sampler derivation."""

        payload = _canonical(self)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return f"sha256:{hashlib.sha256(b'information-set-spec-v1\\0' + encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class _Slot:
    """Internal exact hidden-card destination reconstructed from public constraints."""

    key: tuple[str, int, int]
    kind: HiddenAllocationKind
    age: int | None = None
    color: Color | None = None
    visible_icons: tuple[Icon, ...] = ()

    @property
    def is_special(self) -> bool:
        return self.color is not None or bool(self.visible_icons)


class _Sha256Rng:
    """Small deterministic SHA-256 counter RNG with explicit domain separation."""

    def __init__(self, seed: int | str | bytes, *domain_parts: str) -> None:
        self._seed = _seed_bytes(seed)
        self._domain = b"\0".join(part.encode("ascii") for part in domain_parts)
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
        """Return a uniform integer in ``range(stop)`` using rejection sampling."""

        if stop <= 0:
            raise ValueError("random range must be positive")
        bits = max(1, (stop - 1).bit_length())
        width = (bits + 7) // 8
        limit = 1 << (width * 8)
        accepted = limit - (limit % stop)
        while True:
            value = int.from_bytes(self._bytes(width), "big")
            if value < accepted:
                return value % stop

    def shuffled(self, values: Iterable[CardId]) -> list[CardId]:
        """Return a Fisher-Yates permutation without relying on Python's RNG implementation."""

        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            other = self.randbelow(index + 1)
            result[index], result[other] = result[other], result[index]
        return result


def _seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bool):
        raise ValueError("sampler seed cannot be a boolean")
    if isinstance(seed, int):
        return f"int:{seed}".encode("ascii")
    if isinstance(seed, str):
        return b"str:" + seed.encode("utf-8")
    if isinstance(seed, bytes):
        return b"bytes:" + seed
    raise TypeError("sampler seed must be an int, str, or bytes")


def _canonical(value: object) -> object:
    """Return a canonical JSON-safe representation for deterministic spec derivation."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"information-set data is not canonicalizable: {type(value).__name__}")


def _constraint_key(constraint: HiddenAllocationConstraint) -> tuple[str, int, str, str, int]:
    return (
        constraint.kind.value,
        -1 if constraint.age is None else constraint.age,
        "" if constraint.color is None else constraint.color.value,
        ",".join(icon.value for icon in constraint.visible_icons),
        -1 if constraint.ordinal is None else constraint.ordinal,
    )


def _zone_unknown_ages(
    zone: ZoneObservation, *, label: str, registry: CardRegistry
) -> tuple[int, ...]:
    known = Counter(registry.card(card_id).age for card_id in zone.known_cards)
    values = Counter(zone.values)
    if any(known[age] > values[age] for age in known):
        raise InformationSetSpecError(
            f"{label} known identities do not fit its public age multiset"
        )
    return tuple(sorted((values - known).elements()))


def _visible_icons_for_observation(
    covered: CoveredCardObservation,
    stack: StackObservation,
) -> tuple[Icon, ...]:
    """Validate and return exactly the public exposed icons for a hidden covered card."""

    if stack.splay is SplayDirection.NONE:
        raise InformationSetSpecError(
            "an identity-hidden covered card cannot be unsplayed and counted"
        )
    if covered.card_id is not None:
        return ()
    if covered.age is not None:
        raise InformationSetSpecError("a hidden covered card cannot expose only its age")
    return covered.visible_icons


class InformationSetSpecBuilder:
    """Trusted builder permitted to inspect one stable live paid-turn state."""

    def __init__(self, registry: CardRegistry | None = None) -> None:
        self._registry = registry or load_card_registry()

    def build(self, state: GameState) -> InformationSetSpec:
        """Extract only an audited current-information skeleton from ``state``.

        The state must be a stable ``PLAY``/``TURN_ACTION`` boundary.  In particular, paused
        effects, physical reveals, and effect variables are rejected rather than smuggling a
        partial private runtime into the sampler.
        """

        registry = self._registry
        if state.phase is not GamePhase.PLAY:
            raise UnsupportedInformationSet("information-set sampling supports only PLAY states")
        if state.active_player is None or not 1 <= state.paid_actions_remaining <= 2:
            raise UnsupportedInformationSet("information-set sampling requires an active paid turn")
        if state.pending_effects or state.effect_variables:
            raise UnsupportedInformationSet("information-set sampling requires no pending effects")
        if state.revealed:
            raise UnsupportedInformationSet("information-set sampling requires no physical reveals")
        if state.terminal_result is not None:
            raise UnsupportedInformationSet(
                "information-set sampling cannot inspect terminal states"
            )
        try:
            assert_state_invariants(state, registry)
        except StateInvariantError as error:
            raise InformationSetSpecError(f"live state violates invariants: {error}") from error

        decision = current_decision(state, registry)
        if (
            decision is None
            or decision.kind is not DecisionKind.TURN_ACTION
            or decision.chooser is not state.active_player
            or decision.executor is not state.active_player
            or decision.context is not None
            or decision.source is not None
        ):
            raise UnsupportedInformationSet("state does not expose one stable TURN_ACTION decision")
        observation = observe(state, decision.chooser, registry)
        if observation != decision.observation:
            raise InformationSetSpecError("paid-turn decision observation is stale")
        boundary = public_boundary(state, decision.chooser, decision, observation)
        constraints = self._constraints(observation)
        return InformationSetSpec(
            chooser=decision.chooser,
            observation=observation,
            boundary=boundary,
            legal_actions=decision.legal_actions,
            catalog_fingerprint=registry.data_fingerprint,
            rules_version=state.rules_version,
            information_policy_version=state.information_policy_version,
            monotonic_ids=PublicMonotonicIds(
                state.starting_meld_decision_ids,
                state.next_decision_id,
                state.next_event_id,
                state.next_dogma_action_id,
            ),
            hidden_allocation_constraints=constraints,
        )

    def _constraints(self, observation: GameObservation) -> tuple[HiddenAllocationConstraint, ...]:
        registry = self._registry
        chooser = observation.viewer
        opponent = _other_player(chooser)
        constraints: list[HiddenAllocationConstraint] = []
        for age in range(1, 10):
            constraints.append(
                HiddenAllocationConstraint(HiddenAllocationKind.NORMAL_ACHIEVEMENT, 1, age=age)
            )
        for supply in observation.supplies:
            constraints.append(
                HiddenAllocationConstraint(
                    HiddenAllocationKind.SUPPLY, supply.count, age=supply.age
                )
            )

        observed_opponent = observation.player(opponent)
        for kind, zone in (
            (HiddenAllocationKind.OPPONENT_HAND, observed_opponent.hand),
            (HiddenAllocationKind.OPPONENT_SCORE, observed_opponent.score_pile),
        ):
            for age, count in sorted(
                Counter(_zone_unknown_ages(zone, label=kind.value, registry=registry)).items()
            ):
                if count:
                    constraints.append(HiddenAllocationConstraint(kind, count, age=age))

        for stack in observed_opponent.board:
            if stack.splay is SplayDirection.NONE:
                if stack.top_card_id is not None and stack.covered_count is None:
                    constraints.append(
                        HiddenAllocationConstraint(
                            HiddenAllocationKind.OPPONENT_UNSPLAYED_COVERED,
                            None,
                            color=stack.color,
                        )
                    )
                continue
            if stack.covered_count is None or len(stack.covered_cards) != stack.covered_count:
                raise InformationSetSpecError("splayed stack must expose every covered-card slot")
            for ordinal, covered in enumerate(stack.covered_cards):
                if covered.card_id is None:
                    constraints.append(
                        HiddenAllocationConstraint(
                            HiddenAllocationKind.OPPONENT_SPLAYED_COVERED,
                            1,
                            age=covered.age,
                            color=stack.color,
                            visible_icons=_visible_icons_for_observation(covered, stack),
                            ordinal=ordinal,
                        )
                    )
        constraints.append(HiddenAllocationConstraint(HiddenAllocationKind.REMOVED, None))
        return tuple(sorted(constraints, key=_constraint_key))


class InformationSetSampler:
    """Reconstruct deterministic current-information samples without a live-state input."""

    def __init__(
        self,
        registry: CardRegistry | None = None,
        *,
        seed: int | str | bytes = 0,
        retry_limit: int = 16,
        strict: bool = True,
    ) -> None:
        if retry_limit < 1:
            raise ValueError("sampler retry limit must be at least one")
        _seed_bytes(seed)  # Validate eagerly, before a policy is put into service.
        self._registry = registry or load_card_registry()
        self._seed = seed
        self._retry_limit = retry_limit
        self._strict = strict

    @property
    def retry_limit(self) -> int:
        """Return the fixed number of derived allocation attempts per sample."""

        return self._retry_limit

    def sample(self, spec: InformationSetSpec, *, sample_index: int = 0) -> GameState | None:
        """Return one synthetic sampled state, or ``None`` only in non-strict mode."""

        if sample_index < 0:
            raise ValueError("sample index cannot be negative")
        self._validate_spec(spec)
        return self._sample_validated(spec, sample_index=sample_index, spec_digest=spec.digest)

    def _sample_validated(
        self,
        spec: InformationSetSpec,
        *,
        sample_index: int,
        spec_digest: str,
    ) -> GameState | None:
        """Sample after the shared information-set contract has been validated once."""

        failures: list[str] = []
        for attempt in range(self._retry_limit):
            rng = _Sha256Rng(
                self._seed,
                "sample",
                spec_digest,
                str(sample_index),
                str(attempt),
            )
            try:
                sampled = self._sample_once(spec, rng)
                verify_sampled_state(spec, sampled, self._registry)
                return sampled
            except _SamplingAttemptFailure as error:
                failures.append(str(error))
        failure = SamplingExhausted(
            f"information-set sample {sample_index} exhausted {self._retry_limit} attempts: "
            f"{failures[-1] if failures else 'no allocation branch'}"
        )
        if self._strict:
            raise failure
        return None

    def sample_many(
        self,
        spec: InformationSetSpec,
        count: int,
    ) -> tuple[GameState | None, ...]:
        """Return index-stable independent samples derived from the same policy-owned seed."""

        if count < 0:
            raise ValueError("sample count cannot be negative")
        self._validate_spec(spec)
        spec_digest = spec.digest
        return tuple(
            self._sample_validated(spec, sample_index=index, spec_digest=spec_digest)
            for index in range(count)
        )

    def _validate_spec(self, spec: InformationSetSpec) -> None:
        registry = self._registry
        if spec.catalog_fingerprint != registry.data_fingerprint:
            raise InformationSetSpecError(
                "information-set catalog fingerprint differs from registry"
            )
        if spec.rules_version != RULES_VERSION:
            raise InformationSetSpecError("information-set rules version is unsupported")
        if spec.information_policy_version not in SUPPORTED_INFORMATION_POLICY_VERSIONS:
            raise InformationSetSpecError("information-set policy version is unsupported")
        _validate_observation_shape(spec.observation, registry)
        _validate_achievement_observation(spec.observation)
        _validate_constraints(spec, registry)

    def _sample_once(self, spec: InformationSetSpec, rng: _Sha256Rng) -> GameState:
        registry = self._registry
        fixed = _fixed_cards(spec.observation, registry)
        slots, splayed_geometries = _exact_slots(spec, registry)
        assigned = _allocate_exact_slots(fixed, slots, splayed_geometries, registry, rng)
        remaining = set(registry.by_id) - set(fixed.values()) - set(assigned.values())
        board_cards = _board_cards(spec.observation, fixed, assigned, remaining, registry, rng)
        used_board_cards = {
            card_id
            for player_stacks in board_cards.values()
            for cards in player_stacks.values()
            for card_id in cards
        }
        remaining -= used_board_cards

        players = _players_from_observation(spec.observation, fixed, assigned, board_cards)
        normal_cards = tuple(assigned[("normal", age, 0)] for age in range(1, 10))
        piles = tuple(
            tuple(rng.shuffled(_assigned_supply_cards(assigned, age))) for age in range(1, 11)
        )
        progress = spec.boundary.turn_progress
        counters = TurnCounters(
            tuple(
                PlayerTurnCounters(
                    player_id,
                    tucked=(
                        progress.self_tucked
                        if player_id is spec.chooser
                        else progress.opponent_tucked
                    ),
                    scored=(
                        progress.self_scored
                        if player_id is spec.chooser
                        else progress.opponent_scored
                    ),
                )
                for player_id in PlayerId
            )  # type: ignore[arg-type]
        )
        return GameState(
            supply=SupplyState(piles),
            players=players,
            normal_achievements=NormalAchievementState(normal_cards),
            removed_cards=tuple(sorted(remaining, key=str)),
            phase=GamePhase.PLAY,
            active_player=spec.chooser,
            turn_number=spec.observation.turn_number,
            paid_actions_remaining=spec.observation.paid_actions_remaining,
            turn_counters=counters,
            pending_effects=(),
            effect_variables=(),
            revealed=(),
            starting_meld_decision_ids=spec.monotonic_ids.starting_meld_decision_ids,
            starting_meld_choices=(None, None),
            next_decision_id=spec.monotonic_ids.next_decision_id,
            next_event_id=spec.monotonic_ids.next_event_id,
            next_dogma_action_id=spec.monotonic_ids.next_dogma_action_id,
            setup=_synthetic_setup(registry),
            rules_version=spec.rules_version,
            information_policy_version=spec.information_policy_version,
        )


def _other_player(player_id: PlayerId) -> PlayerId:
    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def _validate_observation_shape(observation: GameObservation, registry: CardRegistry) -> None:
    if tuple(item.age for item in observation.supplies) != tuple(range(1, 11)):
        raise InformationSetSpecError("information-set supplies are not in canonical order")
    if any(item.count < 0 for item in observation.supplies):
        raise InformationSetSpecError("information-set supply count cannot be negative")
    if tuple(player.player_id for player in observation.players) != tuple(PlayerId):
        raise InformationSetSpecError("information-set players are not in canonical order")
    for player in observation.players:
        for zone_name, zone in (("hand", player.hand), ("score", player.score_pile)):
            if any(not 1 <= age <= 10 for age in zone.values):
                raise InformationSetSpecError(
                    f"{player.player_id} {zone_name} has invalid public ages"
                )
            if len(set(zone.known_cards)) != len(zone.known_cards):
                raise InformationSetSpecError(
                    f"{player.player_id} {zone_name} repeats known identities"
                )
            _zone_unknown_ages(zone, label=f"{player.player_id} {zone_name}", registry=registry)
        if tuple(stack.color for stack in player.board) != tuple(Color):
            raise InformationSetSpecError("information-set board colors are not canonical")
        for stack in player.board:
            if stack.top_card_id is None:
                if stack.covered_cards or stack.covered_count not in (0, None):
                    raise InformationSetSpecError("empty stack cannot have covered-card data")
            else:
                card = registry.card(stack.top_card_id)
                if card.color is not stack.color:
                    raise InformationSetSpecError("stack top card has the wrong color")
            if stack.splay is SplayDirection.NONE:
                if stack.covered_count is not None and stack.covered_count != len(
                    stack.covered_cards
                ):
                    raise InformationSetSpecError(
                        "known unsplayed covered count disagrees with cards"
                    )
            elif stack.covered_count is None or len(stack.covered_cards) != stack.covered_count:
                raise InformationSetSpecError("splayed covered-card count is not fully visible")


def _validate_achievement_observation(observation: GameObservation) -> None:
    normal_claimed = tuple(
        achievement for player in observation.players for achievement in player.normal_achievements
    )
    special_claimed = tuple(
        achievement for player in observation.players for achievement in player.special_achievements
    )
    if len(set(normal_claimed)) != len(normal_claimed) or len(set(special_claimed)) != len(
        special_claimed
    ):
        raise InformationSetSpecError("achievement ownership cannot repeat")
    if set(normal_claimed) | set(observation.available_normal_achievements) != set(
        NormalAchievementId
    ):
        raise InformationSetSpecError("normal achievement availability is incomplete")
    from innovation_ai.innovation.types import SpecialAchievementId

    if set(special_claimed) | set(observation.available_special_achievements) != set(
        SpecialAchievementId
    ):
        raise InformationSetSpecError("special achievement availability is incomplete")


def _validate_constraints(spec: InformationSetSpec, registry: CardRegistry) -> None:
    constraints = spec.hidden_allocation_constraints
    if tuple(sorted(constraints, key=_constraint_key)) != constraints:
        raise InformationSetSpecError("hidden allocation constraints are not canonically ordered")
    kinds = Counter(constraint.kind for constraint in constraints)
    if kinds[HiddenAllocationKind.REMOVED] != 1:
        raise InformationSetSpecError(
            "information-set must contain one variable removed-card constraint"
        )
    normal_ages = tuple(
        constraint.age
        for constraint in constraints
        if constraint.kind is HiddenAllocationKind.NORMAL_ACHIEVEMENT
    )
    if normal_ages != tuple(range(1, 10)):
        raise InformationSetSpecError(
            "information-set must preserve one normal achievement per age"
        )
    supplied = {
        constraint.age: constraint.count
        for constraint in constraints
        if constraint.kind is HiddenAllocationKind.SUPPLY
    }
    expected_supplies = {supply.age: supply.count for supply in spec.observation.supplies}
    if supplied != expected_supplies:
        raise InformationSetSpecError("information-set supply constraints differ from observation")
    # Ensure caller-provided specs cannot introduce a hidden identity by naming an impossible icon
    # pattern.  The allocation itself validates every card against the exact splay geometry.
    for constraint in constraints:
        if constraint.color is not None and constraint.visible_icons:
            possible = any(card.color is constraint.color for card in registry.cards)
            if not possible:  # pragma: no cover - catalog invariant, retained for strict input API
                raise InformationSetSpecError("hidden stack constraint names an unknown color")


def _fixed_cards(
    observation: GameObservation, registry: CardRegistry
) -> dict[tuple[str, int, int], CardId]:
    fixed: dict[tuple[str, int, int], CardId] = {}
    seen: dict[CardId, tuple[str, int, int]] = {}

    def add(key: tuple[str, int, int], card_id: CardId) -> None:
        registry.card(card_id)
        if card_id in seen:
            raise InformationSetSpecError(
                f"known card {card_id} occurs in multiple observation slots"
            )
        seen[card_id] = key
        fixed[key] = card_id

    for player in observation.players:
        for zone_name, zone in (("hand", player.hand), ("score", player.score_pile)):
            for ordinal, card_id in enumerate(zone.known_cards):
                if registry.card(card_id).age not in zone.values:
                    raise InformationSetSpecError(
                        f"known {zone_name} card age is absent from public multiset"
                    )
                add((f"{player.player_id.value}-{zone_name}", ordinal, 0), card_id)
        for color_index, stack in enumerate(player.board):
            if stack.top_card_id is not None:
                add((f"{player.player_id.value}-top", color_index, 0), stack.top_card_id)
            for ordinal, covered in enumerate(stack.covered_cards):
                if covered.card_id is not None:
                    card = registry.card(covered.card_id)
                    if card.color is not stack.color:
                        raise InformationSetSpecError(
                            "known covered card has the wrong stack color"
                        )
                    if covered.age is not None and card.age != covered.age:
                        raise InformationSetSpecError(
                            "known covered card age disagrees with observation"
                        )
                    add(
                        (f"{player.player_id.value}-covered", color_index, ordinal), covered.card_id
                    )
    chooser_observation = observation.player(observation.viewer)
    if len(chooser_observation.hand.known_cards) != chooser_observation.hand.count:
        raise InformationSetSpecError("chooser hand must contain no unknown identities")
    if len(chooser_observation.score_pile.known_cards) != chooser_observation.score_pile.count:
        raise InformationSetSpecError("chooser score pile must contain no unknown identities")
    return fixed


def _exact_slots(
    spec: InformationSetSpec,
    registry: CardRegistry,
) -> tuple[tuple[_Slot, ...], dict[tuple[str, int, int], tuple[SplayDirection, tuple[Icon, ...]]]]:
    del registry
    slots: list[_Slot] = []
    geometries: dict[tuple[str, int, int], tuple[SplayDirection, tuple[Icon, ...]]] = {}
    slot_ordinals: defaultdict[tuple[HiddenAllocationKind, int | None], int] = defaultdict(int)
    for constraint in spec.hidden_allocation_constraints:
        if constraint.kind in {
            HiddenAllocationKind.REMOVED,
            HiddenAllocationKind.OPPONENT_UNSPLAYED_COVERED,
        }:
            continue
        assert constraint.count is not None
        for repeat in range(constraint.count):
            if constraint.kind is HiddenAllocationKind.NORMAL_ACHIEVEMENT:
                assert constraint.age is not None
                key = ("normal", constraint.age, repeat)
            elif constraint.kind is HiddenAllocationKind.SUPPLY:
                assert constraint.age is not None
                key = ("supply", constraint.age, repeat)
            elif constraint.kind is HiddenAllocationKind.OPPONENT_HAND:
                assert constraint.age is not None
                index = slot_ordinals[(constraint.kind, constraint.age)]
                slot_ordinals[(constraint.kind, constraint.age)] += 1
                key = ("opponent-hand", constraint.age, index)
            elif constraint.kind is HiddenAllocationKind.OPPONENT_SCORE:
                assert constraint.age is not None
                index = slot_ordinals[(constraint.kind, constraint.age)]
                slot_ordinals[(constraint.kind, constraint.age)] += 1
                key = ("opponent-score", constraint.age, index)
            else:
                assert constraint.kind is HiddenAllocationKind.OPPONENT_SPLAYED_COVERED
                assert constraint.color is not None and constraint.ordinal is not None
                key = ("opponent-covered", tuple(Color).index(constraint.color), constraint.ordinal)
                stack = spec.observation.player(_other_player(spec.chooser)).board[
                    tuple(Color).index(constraint.color)
                ]
                geometries[key] = (stack.splay, constraint.visible_icons)
            slots.append(
                _Slot(
                    key,
                    constraint.kind,
                    age=constraint.age,
                    color=constraint.color,
                    visible_icons=constraint.visible_icons,
                )
            )
    if len({slot.key for slot in slots}) != len(slots):
        raise InformationSetSpecError(
            "hidden allocation constraints duplicate an exact destination"
        )
    return tuple(slots), geometries


def _matches_slot(
    slot: _Slot,
    card_id: CardId,
    geometries: dict[tuple[str, int, int], tuple[SplayDirection, tuple[Icon, ...]]],
    registry: CardRegistry,
) -> bool:
    card = registry.card(card_id)
    if slot.age is not None and card.age != slot.age:
        return False
    if slot.color is not None and card.color is not slot.color:
        return False
    geometry = geometries.get(slot.key)
    if geometry is None:
        return True
    direction, expected = geometry
    actual = tuple(
        icon
        for icon_slot in covered_visible_slots(direction)
        if (icon := card.icon_at(icon_slot)) is not None
    )
    return actual == expected


def _allocate_exact_slots(
    fixed: dict[tuple[str, int, int], CardId],
    slots: tuple[_Slot, ...],
    geometries: dict[tuple[str, int, int], tuple[SplayDirection, tuple[Icon, ...]]],
    registry: CardRegistry,
    rng: _Sha256Rng,
) -> dict[tuple[str, int, int], CardId]:
    available = set(registry.by_id) - set(fixed.values())
    special = tuple(slot for slot in slots if slot.is_special)
    simple = tuple(slot for slot in slots if not slot.is_special)
    required_by_age: Counter[int] = Counter()
    for slot in simple:
        if slot.age is None:
            raise InformationSetSpecError("simple hidden allocation slot lacks an age")
        required_by_age[slot.age] += 1

    def enough_for_simple(cards: set[CardId]) -> bool:
        counts = Counter(registry.card(card_id).age for card_id in cards)
        return all(counts[age] >= count for age, count in required_by_age.items())

    def search(
        remaining_slots: tuple[_Slot, ...],
        cards: set[CardId],
        assigned: dict[tuple[str, int, int], CardId],
    ) -> dict[tuple[str, int, int], CardId] | None:
        if not remaining_slots:
            return assigned if enough_for_simple(cards) else None
        candidate_sets: list[tuple[int, _Slot, list[CardId]]] = []
        for index, slot in enumerate(remaining_slots):
            candidates = sorted(
                (
                    card_id
                    for card_id in cards
                    if _matches_slot(slot, card_id, geometries, registry)
                ),
                key=str,
            )
            if not candidates:
                return None
            candidate_sets.append((index, slot, candidates))
        index, slot, candidates = min(
            candidate_sets,
            key=lambda item: (len(item[2]), item[1].key),
        )
        for card_id in rng.shuffled(candidates):
            next_cards = set(cards)
            next_cards.remove(card_id)
            if not enough_for_simple(next_cards):
                continue
            next_assigned = dict(assigned)
            next_assigned[slot.key] = card_id
            result = search(
                (*remaining_slots[:index], *remaining_slots[index + 1 :]),
                next_cards,
                next_assigned,
            )
            if result is not None:
                return result
        return None

    assigned = search(special, available, {})
    if assigned is None:
        raise _SamplingAttemptFailure("no hidden splayed-board assignment satisfies public icons")
    remaining = available - set(assigned.values())
    if not enough_for_simple(remaining):
        raise _SamplingAttemptFailure("hidden exact-age slots exceed remaining age capacities")
    by_age: dict[int, list[_Slot]] = defaultdict(list)
    for slot in simple:
        assert slot.age is not None
        by_age[slot.age].append(slot)
    for age in sorted(by_age):
        candidates = sorted(
            (card_id for card_id in remaining if registry.card(card_id).age == age),
            key=str,
        )
        randomized = rng.shuffled(candidates)
        slots_for_age = sorted(by_age[age], key=lambda slot: slot.key)
        if len(randomized) < len(slots_for_age):
            raise _SamplingAttemptFailure("not enough cards for exact-age hidden slots")
        for slot, card_id in zip(slots_for_age, randomized, strict=False):
            assigned[slot.key] = card_id
            remaining.remove(card_id)
    return assigned


def _board_cards(
    observation: GameObservation,
    fixed: dict[tuple[str, int, int], CardId],
    assigned: dict[tuple[str, int, int], CardId],
    remaining: set[CardId],
    registry: CardRegistry,
    rng: _Sha256Rng,
) -> dict[PlayerId, dict[Color, tuple[CardId, ...]]]:
    result: dict[PlayerId, dict[Color, tuple[CardId, ...]]] = {}
    chooser = observation.viewer
    opponent = _other_player(chooser)
    for player in observation.players:
        stacks: dict[Color, tuple[CardId, ...]] = {}
        for color_index, stack in enumerate(player.board):
            top = (
                None
                if stack.top_card_id is None
                else fixed[(f"{player.player_id.value}-top", color_index, 0)]
            )
            known_or_assigned: list[CardId] = []
            if stack.splay is not SplayDirection.NONE or stack.covered_count is not None:
                for ordinal, covered in enumerate(stack.covered_cards):
                    if covered.card_id is not None:
                        known_or_assigned.append(
                            fixed[(f"{player.player_id.value}-covered", color_index, ordinal)]
                        )
                    elif player.player_id is opponent:
                        known_or_assigned.append(
                            assigned[("opponent-covered", color_index, ordinal)]
                        )
                    else:
                        raise InformationSetSpecError(
                            "chooser board contains an unknown covered card"
                        )
            elif top is not None and player.player_id is opponent:
                candidates = sorted(
                    (
                        card_id
                        for card_id in remaining
                        if registry.card(card_id).color is stack.color
                    ),
                    key=str,
                )
                take = rng.randbelow(len(candidates) + 1)
                selected = rng.shuffled(candidates)[:take]
                known_or_assigned.extend(selected)
                remaining.difference_update(selected)
            elif stack.covered_cards:
                raise InformationSetSpecError("unknown covered-card shape is unsupported")
            if top is None:
                stacks[stack.color] = ()
            else:
                stacks[stack.color] = (*known_or_assigned, top)
        result[player.player_id] = stacks
    return result


def _players_from_observation(
    observation: GameObservation,
    fixed: dict[tuple[str, int, int], CardId],
    assigned: dict[tuple[str, int, int], CardId],
    board_cards: dict[PlayerId, dict[Color, tuple[CardId, ...]]],
) -> tuple[PlayerState, PlayerState]:
    chooser = observation.viewer
    opponent = _other_player(chooser)
    players: list[PlayerState] = []
    for player in observation.players:
        hand = list(player.hand.known_cards)
        score = list(player.score_pile.known_cards)
        if player.player_id is opponent:
            hand.extend(card_id for key, card_id in assigned.items() if key[0] == "opponent-hand")
            score.extend(card_id for key, card_id in assigned.items() if key[0] == "opponent-score")
        elif player.player_id is chooser:
            # These are already checked as complete in ``_fixed_cards``.
            pass
        else:  # pragma: no cover - two-player enum guard
            raise AssertionError("unexpected player relation")
        stacks = tuple(
            ColorStack(
                color,
                board_cards[player.player_id][color],
                player.board[tuple(Color).index(color)].splay
                if len(board_cards[player.player_id][color]) >= 2
                else SplayDirection.NONE,
            )
            for color in Color
        )
        players.append(
            PlayerState(
                player.player_id,
                tuple(hand),
                Board(stacks),
                tuple(score),
                player.normal_achievements,
                player.special_achievements,
            )
        )
    return players[0], players[1]


def _assigned_supply_cards(
    assigned: dict[tuple[str, int, int], CardId], age: int
) -> tuple[CardId, ...]:
    return tuple(
        card_id for key, card_id in sorted(assigned.items()) if key[0] == "supply" and key[1] == age
    )


def _synthetic_setup(registry: CardRegistry) -> SetupProvenance:
    """Return fixed synthetic provenance, never copied from a sampled game's true setup."""

    return SetupProvenance(
        SYNTHETIC_SETUP_SEED,
        registry.data_fingerprint,
        tuple(
            tuple(sorted((card.id for card in registry.cards if card.age == age), key=str))
            for age in range(1, 11)
        ),
        tuple(player_id for _ in range(2) for player_id in PlayerId),
        rng_version=SYNTHETIC_SETUP_RNG_VERSION,
    )


def verify_sampled_state(
    spec: InformationSetSpec,
    sampled: GameState,
    registry: CardRegistry | None = None,
) -> None:
    """Require a synthetic state to reproduce every audited current-information boundary."""

    registry = registry or load_card_registry()
    if sampled.setup != _synthetic_setup(registry):
        raise SampleVerificationError("sample retained non-synthetic setup provenance")
    try:
        assert_state_invariants(sampled, registry)
    except StateInvariantError as error:
        raise SampleVerificationError(f"sample violates state invariants: {error}") from error
    observation = observe(sampled, spec.chooser, registry)
    if observation != spec.observation:
        raise SampleVerificationError("sample does not reproduce chooser observation")
    decision = current_decision(sampled, registry)
    if decision is None or decision.kind is not DecisionKind.TURN_ACTION:
        raise SampleVerificationError("sample does not expose a turn-action decision")
    if decision.legal_actions != spec.legal_actions:
        raise SampleVerificationError("sample does not reproduce semantic legal actions")
    boundary = public_boundary(sampled, spec.chooser, decision, observation)
    if boundary != spec.boundary:
        raise SampleVerificationError("sample does not reproduce public turn progress/boundary")


__all__ = [
    "INFORMATION_SET_RNG_VERSION",
    "INFORMATION_SET_SAMPLER_VERSION",
    "INFORMATION_SET_SPEC_SCHEMA_VERSION",
    "SYNTHETIC_SETUP_RNG_VERSION",
    "SYNTHETIC_SETUP_SEED",
    "HiddenAllocationConstraint",
    "HiddenAllocationKind",
    "InformationSetError",
    "InformationSetSampler",
    "InformationSetSpec",
    "InformationSetSpecBuilder",
    "InformationSetSpecError",
    "PublicMonotonicIds",
    "SampleVerificationError",
    "SamplingError",
    "SamplingExhausted",
    "UnsupportedInformationSet",
    "verify_sampled_state",
]
