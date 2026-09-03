"""Encoder-independent, compact, hash-verifiable training replay episodes.

Compact replays deliberately retain setup provenance and semantic actions, but omit decisions,
observations, and transition hashes.  They are a durable source format: a verifier reconstructs
normal engine play and rejects incompatible or divergent episodes before an encoder sees them.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from innovation_ai.innovation.actions import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    Decision,
    SemanticAction,
    action_payload,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.registry import effects_fingerprint
from innovation_ai.innovation.invariants import InvariantViolation, assert_state_properties
from innovation_ai.innovation.logs import ENGINE_VERSION, ReplayOutcome
from innovation_ai.innovation.observations import OBSERVATION_SCHEMA_VERSION
from innovation_ai.innovation.protocol import InnovationEngineError
from innovation_ai.innovation.replay import (
    DefaultReplayAdapter,
    ReplayAdapter,
    _decision_for_action,
)
from innovation_ai.innovation.serialization import (
    SerializationError,
    action_from_payload,
    setup_from_payload,
    setup_payload,
    terminal_from_payload,
    terminal_payload,
)
from innovation_ai.innovation.state import (
    RULES_VERSION,
    SETUP_RNG_VERSION,
    STATE_SCHEMA_VERSION,
    SUPPORTED_INFORMATION_POLICY_VERSIONS,
    TERMINAL_SCHEMA_VERSION,
    GameState,
    SetupProvenance,
    TerminalResult,
    state_hash,
)
from innovation_ai.innovation.types import PlayerId

COMPACT_REPLAY_FORMAT = "innovation-ai-compact-episode"
COMPACT_REPLAY_SCHEMA_VERSION = 1
COMPACT_REPLAY_MANIFEST_FORMAT = "innovation-ai-compact-replay-manifest"
COMPACT_REPLAY_MANIFEST_SCHEMA_VERSION = 1

_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class CompactReplayError(ValueError):
    """Base class for compact replay parsing, recording, and shard failures."""


class CompactReplaySchemaError(CompactReplayError):
    """A compact replay document does not satisfy its strict schema."""


class CompactReplayCompatibilityError(CompactReplayError):
    """A valid compact replay cannot be interpreted by this engine/catalog."""


class CompactReplayDivergenceError(CompactReplayError):
    """Reconstructed setup/actions differ from a compact episode's final markers."""


class CompactReplayRecordingError(CompactReplayError):
    """A caller attempted to record outside a replayable setup/action boundary."""


class CompactReplayShardError(CompactReplayError):
    """A compact replay shard or its preassigned manifest is invalid."""


def canonical_json(payload: JsonValue) -> str:
    """Encode a finite JSON value in the one permitted compact-replay representation."""

    _validate_json_value(payload, "json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def parse_canonical_json(text: str) -> JsonValue:
    """Parse canonical JSON, rejecting duplicate keys, non-finite numbers, and alternate bytes."""

    def reject_constant(value: str) -> None:
        raise CompactReplaySchemaError(f"non-finite JSON constant {value!r} is not permitted")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CompactReplaySchemaError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise CompactReplaySchemaError(f"invalid JSON: {error}") from error
    _validate_json_value(raw, "json")
    value = cast(JsonValue, raw)
    if canonical_json(value) != text:
        raise CompactReplaySchemaError("JSON is not in canonical compact-replay form")
    return value


def sha256_digest(value: bytes | str | JsonValue) -> str:
    """Return a tagged SHA-256 digest of bytes, UTF-8 text, or canonical JSON."""

    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = canonical_json(value).encode("ascii")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def setup_provenance_digest(setup: SetupProvenance) -> str:
    """Digest every setup input, including explicit shuffled piles, for split grouping."""

    return sha256_digest(cast(JsonValue, setup_payload(setup)))


@dataclass(frozen=True, slots=True)
class SeatPolicyProvenance:
    """One canonical seat's complete producer policy identity.

    ``checkpoint_id`` is ``None`` for non-checkpoint policies such as a heuristic fallback.
    ``policy_descriptor_id`` remains mandatory because it identifies the complete policy settings,
    not merely model weights.
    """

    seat: PlayerId
    policy_descriptor_id: str
    checkpoint_id: str | None
    agent_rng_version: str

    def __post_init__(self) -> None:
        _required_text(self.policy_descriptor_id, "policy descriptor ID")
        _required_text(self.agent_rng_version, "agent RNG version")
        if self.checkpoint_id is not None:
            _required_text(self.checkpoint_id, "checkpoint ID")


@dataclass(frozen=True, slots=True)
class ExplorationProvenance:
    """Versioned selection/exploration settings used to produce an episode."""

    selector_version: str
    temperature: float
    rng_version: str

    def __post_init__(self) -> None:
        _required_text(self.selector_version, "selector version")
        _required_text(self.rng_version, "exploration RNG version")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("exploration temperature must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DeterminizationProvenance:
    """Versioned hidden-state-sampling configuration used by an actor.

    ``search_descriptor_id`` is absent from legacy payloads.  Milestone-4 producers include it
    when sampled search can make any decision in the episode.
    """

    sampler_version: str
    rng_version: str
    count: int
    failure_policy_id: str | None
    strict: bool
    search_descriptor_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.sampler_version, "sampler version")
        _required_text(self.rng_version, "determinization RNG version")
        if self.count < 0:
            raise ValueError("determinizations count cannot be negative")
        if self.failure_policy_id is not None:
            _required_text(self.failure_policy_id, "determinization failure policy ID")
        if self.search_descriptor_id is not None:
            _digest(self.search_descriptor_id, "search descriptor ID")


@dataclass(frozen=True, slots=True)
class CompactReplayProvenance:
    """Actor metadata shared by every episode in a producer run."""

    producer_run_id: str
    resolved_config_digest: str
    generation: int
    seat_mapping: tuple[SeatPolicyProvenance, SeatPolicyProvenance]
    exploration: ExplorationProvenance
    determinization: DeterminizationProvenance

    def __post_init__(self) -> None:
        _required_text(self.producer_run_id, "producer run ID")
        _digest(self.resolved_config_digest, "resolved configuration digest")
        if self.generation < 0:
            raise ValueError("generation cannot be negative")
        if tuple(item.seat for item in self.seat_mapping) != tuple(PlayerId):
            raise ValueError("seat mapping must contain player-1 then player-2 exactly once")


@dataclass(frozen=True, slots=True)
class CompactEpisode:
    """Immutable compact terminal episode with no decisions, observations, or transition hashes."""

    episode_id: str
    engine_version: str
    rules_version: str
    information_policy_version: str
    card_data_fingerprint: str
    effects_fingerprint: str
    setup: SetupProvenance
    provenance: CompactReplayProvenance
    actions: tuple[SemanticAction, ...]
    transition_count: int
    terminal_result: TerminalResult
    final_state_hash: str
    format: str = COMPACT_REPLAY_FORMAT
    schema_version: int = COMPACT_REPLAY_SCHEMA_VERSION
    state_schema_version: int = STATE_SCHEMA_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    decision_schema_version: int = DECISION_SCHEMA_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    terminal_schema_version: int = TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _EPISODE_ID.fullmatch(self.episode_id) is None:
            raise ValueError(f"invalid compact episode ID {self.episode_id!r}")
        if self.format != COMPACT_REPLAY_FORMAT:
            raise ValueError(f"unsupported compact replay format {self.format!r}")
        if self.schema_version != COMPACT_REPLAY_SCHEMA_VERSION:
            raise ValueError(f"unsupported compact replay schema version {self.schema_version}")
        if self.transition_count != len(self.actions):
            raise ValueError("compact episode transition count does not match actions")
        if self.setup.card_data_fingerprint != self.card_data_fingerprint:
            raise ValueError("compact episode setup and header card fingerprints differ")
        _digest(self.card_data_fingerprint, "card-data fingerprint")
        _digest(self.effects_fingerprint, "effects fingerprint")
        _digest(self.final_state_hash, "final state hash")
        for name, actual, expected in _schema_versions(self):
            if actual != expected:
                raise ValueError(f"unsupported compact episode {name} version {actual}")

    @property
    def setup_digest(self) -> str:
        """Return the content digest used for episode-level dataset split grouping."""

        return setup_provenance_digest(self.setup)


@dataclass(frozen=True, slots=True)
class CompactReplayShardManifest:
    """Preassigned episode membership for one deterministic compact JSONL shard."""

    shard_id: str
    episode_ids: tuple[str, ...]
    format: str = COMPACT_REPLAY_MANIFEST_FORMAT
    schema_version: int = COMPACT_REPLAY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _EPISODE_ID.fullmatch(self.shard_id) is None:
            raise ValueError(f"invalid compact shard ID {self.shard_id!r}")
        if self.format != COMPACT_REPLAY_MANIFEST_FORMAT:
            raise ValueError(f"unsupported compact replay manifest format {self.format!r}")
        if self.schema_version != COMPACT_REPLAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported compact replay manifest schema version")
        if not self.episode_ids:
            raise ValueError("compact replay manifest cannot be empty")
        if tuple(sorted(self.episode_ids)) != self.episode_ids:
            raise ValueError("compact replay manifest episode IDs must be canonical sorted order")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("compact replay manifest episode IDs must be unique")
        for episode_id in self.episode_ids:
            if _EPISODE_ID.fullmatch(episode_id) is None:
                raise ValueError(f"invalid compact episode ID {episode_id!r}")


def compact_episode_payload(episode: CompactEpisode) -> JsonObject:
    """Return the complete canonical, actions-only compact episode payload."""

    return {
        "format": episode.format,
        "schema_version": episode.schema_version,
        "episode_id": episode.episode_id,
        "engine_version": episode.engine_version,
        "rules_version": episode.rules_version,
        "information_policy_version": episode.information_policy_version,
        "state_schema_version": episode.state_schema_version,
        "action_schema_version": episode.action_schema_version,
        "decision_schema_version": episode.decision_schema_version,
        "observation_schema_version": episode.observation_schema_version,
        "terminal_schema_version": episode.terminal_schema_version,
        "card_data_fingerprint": episode.card_data_fingerprint,
        "effects_fingerprint": episode.effects_fingerprint,
        "setup": cast(JsonValue, setup_payload(episode.setup)),
        "generation": episode.provenance.generation,
        "seat_mapping": [_seat_payload(item) for item in episode.provenance.seat_mapping],
        "exploration": _exploration_payload(episode.provenance.exploration),
        "determinization": _determinization_payload(episode.provenance.determinization),
        "actions": [cast(JsonValue, action_payload(action)) for action in episode.actions],
        "transition_count": episode.transition_count,
        "terminal_result": cast(JsonValue, terminal_payload(episode.terminal_result)),
        "final_state_hash": episode.final_state_hash,
        "producer_run_id": episode.provenance.producer_run_id,
        "resolved_config_digest": episode.provenance.resolved_config_digest,
    }


def dumps_compact_episode(episode: CompactEpisode) -> str:
    """Serialize one compact episode to deterministic one-line canonical JSON."""

    return canonical_json(cast(JsonValue, compact_episode_payload(episode)))


def loads_compact_episode(text: str) -> CompactEpisode:
    """Parse exactly one canonical compact episode without replaying its actions."""

    return compact_episode_from_payload(parse_canonical_json(text))


def compact_episode_from_payload(value: object) -> CompactEpisode:
    """Decode a strict compact episode payload, rejecting missing or unknown fields."""

    payload = _object(value, "compact_episode")
    _exact_keys(payload, _EPISODE_KEYS, "compact_episode")
    try:
        seat_mapping = tuple(
            _seat_from_payload(item, "compact_episode.seat_mapping[]")
            for item in _array(payload["seat_mapping"], "compact_episode.seat_mapping")
        )
        if len(seat_mapping) != len(PlayerId):
            raise CompactReplaySchemaError("compact_episode.seat_mapping must contain two seats")
        provenance = CompactReplayProvenance(
            producer_run_id=_string(payload["producer_run_id"], "compact_episode.producer_run_id"),
            resolved_config_digest=_string(
                payload["resolved_config_digest"], "compact_episode.resolved_config_digest"
            ),
            generation=_integer(payload["generation"], "compact_episode.generation"),
            seat_mapping=cast(tuple[SeatPolicyProvenance, SeatPolicyProvenance], seat_mapping),
            exploration=_exploration_from_payload(payload["exploration"]),
            determinization=_determinization_from_payload(payload["determinization"]),
        )
        return CompactEpisode(
            episode_id=_string(payload["episode_id"], "compact_episode.episode_id"),
            engine_version=_string(payload["engine_version"], "compact_episode.engine_version"),
            rules_version=_string(payload["rules_version"], "compact_episode.rules_version"),
            information_policy_version=_string(
                payload["information_policy_version"], "compact_episode.information_policy_version"
            ),
            card_data_fingerprint=_string(
                payload["card_data_fingerprint"], "compact_episode.card_data_fingerprint"
            ),
            effects_fingerprint=_string(
                payload["effects_fingerprint"], "compact_episode.effects_fingerprint"
            ),
            setup=setup_from_payload(payload["setup"]),
            provenance=provenance,
            actions=tuple(
                action_from_payload(item)
                for item in _array(payload["actions"], "compact_episode.actions")
            ),
            transition_count=_integer(
                payload["transition_count"], "compact_episode.transition_count"
            ),
            terminal_result=terminal_from_payload(payload["terminal_result"]),
            final_state_hash=_string(
                payload["final_state_hash"], "compact_episode.final_state_hash"
            ),
            format=_string(payload["format"], "compact_episode.format"),
            schema_version=_integer(payload["schema_version"], "compact_episode.schema_version"),
            state_schema_version=_integer(
                payload["state_schema_version"], "compact_episode.state_schema_version"
            ),
            action_schema_version=_integer(
                payload["action_schema_version"], "compact_episode.action_schema_version"
            ),
            decision_schema_version=_integer(
                payload["decision_schema_version"], "compact_episode.decision_schema_version"
            ),
            observation_schema_version=_integer(
                payload["observation_schema_version"], "compact_episode.observation_schema_version"
            ),
            terminal_schema_version=_integer(
                payload["terminal_schema_version"], "compact_episode.terminal_schema_version"
            ),
        )
    except (SerializationError, TypeError, ValueError) as error:
        if isinstance(error, CompactReplayError):
            raise
        raise CompactReplaySchemaError(f"invalid compact episode: {error}") from error


def compact_replay_manifest_payload(manifest: CompactReplayShardManifest) -> JsonObject:
    """Return the canonical preassignment manifest payload."""

    return {
        "format": manifest.format,
        "schema_version": manifest.schema_version,
        "shard_id": manifest.shard_id,
        "episode_ids": list(manifest.episode_ids),
    }


def dumps_compact_replay_manifest(manifest: CompactReplayShardManifest) -> str:
    """Serialize a preassigned shard manifest in canonical JSON."""

    return canonical_json(cast(JsonValue, compact_replay_manifest_payload(manifest)))


def loads_compact_replay_manifest(text: str) -> CompactReplayShardManifest:
    """Parse a strict canonical preassigned shard manifest."""

    payload = _object(parse_canonical_json(text), "compact_replay_manifest")
    _exact_keys(payload, {"format", "schema_version", "shard_id", "episode_ids"}, "manifest")
    try:
        return CompactReplayShardManifest(
            shard_id=_string(payload["shard_id"], "manifest.shard_id"),
            episode_ids=tuple(
                _string(item, "manifest.episode_ids[]")
                for item in _array(payload["episode_ids"], "manifest.episode_ids")
            ),
            format=_string(payload["format"], "manifest.format"),
            schema_version=_integer(payload["schema_version"], "manifest.schema_version"),
        )
    except ValueError as error:
        raise CompactReplaySchemaError(f"invalid compact replay manifest: {error}") from error


def compact_episode_digest(episode: CompactEpisode) -> str:
    """Return the content digest of a canonical compact episode document."""

    return sha256_digest(dumps_compact_episode(episode))


@dataclass(frozen=True, slots=True)
class VerifiedCompactEpisode:
    """Terminal state obtained by independently reconstructing a compact episode."""

    episode: CompactEpisode
    state: GameState
    transitions_replayed: int


def check_compact_episode_compatibility(
    episode: CompactEpisode, registry: CardRegistry | None = None
) -> None:
    """Reject versions, fingerprints, and setup conventions unsupported by this engine."""

    registry = registry or load_card_registry()
    expected: tuple[tuple[str, object, object], ...] = (
        ("format", episode.format, COMPACT_REPLAY_FORMAT),
        ("compact replay schema", episode.schema_version, COMPACT_REPLAY_SCHEMA_VERSION),
        ("engine", episode.engine_version, ENGINE_VERSION),
        ("rules", episode.rules_version, RULES_VERSION),
        ("state schema", episode.state_schema_version, STATE_SCHEMA_VERSION),
        ("action schema", episode.action_schema_version, ACTION_SCHEMA_VERSION),
        ("decision schema", episode.decision_schema_version, DECISION_SCHEMA_VERSION),
        ("observation schema", episode.observation_schema_version, OBSERVATION_SCHEMA_VERSION),
        ("terminal schema", episode.terminal_schema_version, TERMINAL_SCHEMA_VERSION),
        ("setup RNG", episode.setup.rng_version, SETUP_RNG_VERSION),
        ("card-data fingerprint", episode.card_data_fingerprint, registry.data_fingerprint),
        (
            "setup card-data fingerprint",
            episode.setup.card_data_fingerprint,
            registry.data_fingerprint,
        ),
        ("effects fingerprint", episode.effects_fingerprint, effects_fingerprint()),
    )
    for name, actual, required in expected:
        if actual != required:
            raise CompactReplayCompatibilityError(
                f"incompatible {name}: episode has {actual!r}, engine expects {required!r}"
            )
    if episode.information_policy_version not in SUPPORTED_INFORMATION_POLICY_VERSIONS:
        raise CompactReplayCompatibilityError(
            "incompatible information policy: "
            f"episode has {episode.information_policy_version!r}, "
            f"engine supports {sorted(SUPPORTED_INFORMATION_POLICY_VERSIONS)!r}"
        )


def verify_compact_episode(
    episode: CompactEpisode,
    registry: CardRegistry | None = None,
    *,
    adapter: ReplayAdapter | None = None,
) -> VerifiedCompactEpisode:
    """Replay setup and actions, rejecting illegal, truncated, edited, or divergent episodes."""

    registry = registry or load_card_registry()
    check_compact_episode_compatibility(episode, registry)
    selected_adapter = adapter or DefaultReplayAdapter(episode.information_policy_version)
    try:
        state = selected_adapter.initial_state(episode.setup, registry)
        assert_state_properties(state, registry)
    except (ValueError, InvariantViolation) as error:
        raise CompactReplayDivergenceError(
            f"initial setup reconstruction failed: {error}"
        ) from error

    for sequence, action in enumerate(episode.actions, start=1):
        decisions = selected_adapter.decisions(state, registry)
        try:
            decision = _decision_for_action(decisions, action, sequence)
        except Exception as error:
            raise CompactReplayDivergenceError(str(error)) from error
        if action not in decision.legal_actions:
            raise CompactReplayDivergenceError(
                f"transition {sequence}: recorded action is not legal"
            )
        try:
            state = selected_adapter.apply(state, action, registry)
            assert_state_properties(state, registry)
        except (InnovationEngineError, ValueError, InvariantViolation) as error:
            raise CompactReplayDivergenceError(
                f"transition {sequence}: action application failed: {error}"
            ) from error

    if selected_adapter.outcome(state) is not ReplayOutcome.TERMINAL:
        raise CompactReplayDivergenceError(
            "episode is truncated: actions did not reach a terminal state"
        )
    if state.terminal_result != episode.terminal_result:
        raise CompactReplayDivergenceError("recorded terminal result differs from replayed state")
    actual_final_hash = state_hash(state)
    if actual_final_hash != episode.final_state_hash:
        raise CompactReplayDivergenceError(
            "final state hash differs: "
            f"recorded {episode.final_state_hash}, reproduced {actual_final_hash}"
        )
    return VerifiedCompactEpisode(episode, state, len(episode.actions))


class CompactReplayRecorder:
    """Actor-side semantic-action recorder that never stores decisions or transition hashes."""

    def __init__(
        self,
        initial_state: GameState,
        episode_id: str,
        provenance: CompactReplayProvenance,
        registry: CardRegistry | None = None,
        *,
        adapter: ReplayAdapter | None = None,
    ) -> None:
        self._registry = registry or load_card_registry()
        self._adapter = adapter or DefaultReplayAdapter(initial_state.information_policy_version)
        if initial_state.setup.card_data_fingerprint != self._registry.data_fingerprint:
            raise CompactReplayRecordingError(
                "initial state's card-data fingerprint is incompatible"
            )
        try:
            reconstructed = self._adapter.initial_state(initial_state.setup, self._registry)
        except ValueError as error:
            raise CompactReplayRecordingError(
                f"could not reconstruct initial setup: {error}"
            ) from error
        if state_hash(reconstructed) != state_hash(initial_state):
            raise CompactReplayRecordingError("recorder must start at the explicit setup boundary")
        if _EPISODE_ID.fullmatch(episode_id) is None:
            raise CompactReplayRecordingError(f"invalid compact episode ID {episode_id!r}")
        self._initial_state = initial_state
        self._state = initial_state
        self._episode_id = episode_id
        self._provenance = provenance
        self._actions: list[SemanticAction] = []

    @property
    def state(self) -> GameState:
        """Return the recorder's current immutable authoritative state."""

        return self._state

    @property
    def actions(self) -> tuple[SemanticAction, ...]:
        """Return the recorded semantic-action prefix for failure diagnostics."""

        return tuple(self._actions)

    def decisions(self) -> tuple[Decision, ...]:
        """Return currently pending semantic decisions for external policy routing."""

        return self._adapter.decisions(self._state, self._registry)

    def submit(self, action: SemanticAction) -> GameState:
        """Validate and record one semantic action, retaining no decision payload or step hash."""

        sequence = len(self._actions) + 1
        try:
            decision = _decision_for_action(self.decisions(), action, sequence)
        except Exception as error:
            raise CompactReplayRecordingError(str(error)) from error
        if action not in decision.legal_actions:
            raise CompactReplayRecordingError(
                f"action is not legal for pending decision {decision.decision_id}"
            )
        try:
            new_state = self._adapter.apply(self._state, action, self._registry)
        except (InnovationEngineError, ValueError) as error:
            raise CompactReplayRecordingError(f"could not record action: {error}") from error
        self._actions.append(action)
        self._state = new_state
        return new_state

    def episode(self) -> CompactEpisode:
        """Freeze a terminal compact episode; nonterminal recordings cannot be sealed."""

        if self._adapter.outcome(self._state) is not ReplayOutcome.TERMINAL:
            raise CompactReplayRecordingError("compact episode can only seal at a terminal state")
        terminal = self._state.terminal_result
        if terminal is None:  # pragma: no cover - adapter contract defense
            raise CompactReplayRecordingError("terminal recorder state lacks a terminal result")
        return CompactEpisode(
            episode_id=self._episode_id,
            engine_version=ENGINE_VERSION,
            rules_version=self._initial_state.rules_version,
            information_policy_version=self._initial_state.information_policy_version,
            card_data_fingerprint=self._registry.data_fingerprint,
            effects_fingerprint=effects_fingerprint(),
            setup=self._initial_state.setup,
            provenance=self._provenance,
            actions=tuple(self._actions),
            transition_count=len(self._actions),
            terminal_result=terminal,
            final_state_hash=state_hash(self._state),
        )

    compact_episode = episode


CompactEpisodeRecorder = CompactReplayRecorder


class CompactReplayShardWriter:
    """Collect a preassigned set of episodes and atomically seal canonical gzip JSONL bytes."""

    def __init__(self, path: Path, manifest: CompactReplayShardManifest) -> None:
        self._path = path
        self._manifest = manifest
        self._episodes: dict[str, CompactEpisode] = {}
        self._sealed = False

    @property
    def manifest(self) -> CompactReplayShardManifest:
        """Return immutable preassigned shard membership."""

        return self._manifest

    def add(self, episode: CompactEpisode) -> None:
        """Accept one assigned episode, independent of its completion order."""

        if self._sealed:
            raise CompactReplayShardError("cannot add an episode after sealing")
        if episode.episode_id not in self._manifest.episode_ids:
            raise CompactReplayShardError(
                "episode "
                f"{episode.episode_id!r} is not assigned to shard "
                f"{self._manifest.shard_id!r}"
            )
        if episode.episode_id in self._episodes:
            raise CompactReplayShardError(f"episode {episode.episode_id!r} was added twice")
        self._episodes[episode.episode_id] = episode

    def seal(self) -> str:
        """Write one deterministic gzip member via atomic rename and return its file digest."""

        if self._sealed:
            raise CompactReplayShardError("compact replay shard is already sealed")
        missing = tuple(
            episode_id
            for episode_id in self._manifest.episode_ids
            if episode_id not in self._episodes
        )
        if missing:
            raise CompactReplayShardError(
                f"cannot seal shard with missing episodes {list(missing)!r}"
            )
        encoded = "".join(
            dumps_compact_episode(self._episodes[episode_id]) + "\n"
            for episode_id in self._manifest.episode_ids
        ).encode("ascii")
        compressed = _deterministic_gzip(encoded)
        _atomic_write(self._path, compressed)
        self._sealed = True
        return sha256_digest(compressed)


def write_compact_replay_shard(
    path: Path,
    manifest: CompactReplayShardManifest,
    episodes: tuple[CompactEpisode, ...] | list[CompactEpisode],
) -> str:
    """Write a completed assigned shard, canonicalizing episode completion order."""

    writer = CompactReplayShardWriter(path, manifest)
    for episode in episodes:
        writer.add(episode)
    return writer.seal()


def read_compact_replay_shard(
    path: Path,
    manifest: CompactReplayShardManifest | None = None,
    *,
    verify: bool = False,
    registry: CardRegistry | None = None,
) -> tuple[CompactEpisode, ...]:
    """Read one fixed-header gzip JSONL shard and optionally replay-verify every episode."""

    try:
        compressed = path.read_bytes()
    except OSError as error:
        raise CompactReplayShardError(
            f"could not read compact replay shard {path}: {error}"
        ) from error
    raw = _read_fixed_gzip(compressed)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise CompactReplayShardError(
            "compact replay JSONL must be ASCII canonical JSON"
        ) from error
    if not text or not text.endswith("\n"):
        raise CompactReplayShardError("compact replay shard must end in exactly one JSONL newline")
    lines = text[:-1].split("\n")
    if any(not line for line in lines):
        raise CompactReplayShardError("compact replay shard contains an empty JSONL record")
    try:
        episodes = tuple(loads_compact_episode(line) for line in lines)
    except CompactReplayError as error:
        raise CompactReplayShardError(f"invalid compact replay JSONL record: {error}") from error
    identifiers = tuple(episode.episode_id for episode in episodes)
    if tuple(sorted(identifiers)) != identifiers or len(set(identifiers)) != len(identifiers):
        raise CompactReplayShardError(
            "compact replay shard episodes are not unique canonical ID order"
        )
    if manifest is not None and identifiers != manifest.episode_ids:
        raise CompactReplayShardError(
            "compact replay shard does not match its preassigned manifest"
        )
    if verify:
        for episode in episodes:
            verify_compact_episode(episode, registry)
    return episodes


def _seat_payload(value: SeatPolicyProvenance) -> dict[str, JsonValue]:
    return {
        "seat": value.seat.value,
        "policy_descriptor_id": value.policy_descriptor_id,
        "checkpoint_id": value.checkpoint_id,
        "agent_rng_version": value.agent_rng_version,
    }


def _seat_from_payload(value: JsonValue, path: str) -> SeatPolicyProvenance:
    payload = _object(value, path)
    _exact_keys(
        payload, {"seat", "policy_descriptor_id", "checkpoint_id", "agent_rng_version"}, path
    )
    raw_checkpoint = payload["checkpoint_id"]
    return SeatPolicyProvenance(
        seat=_player_id(payload["seat"], f"{path}.seat"),
        policy_descriptor_id=_string(
            payload["policy_descriptor_id"], f"{path}.policy_descriptor_id"
        ),
        checkpoint_id=(
            None if raw_checkpoint is None else _string(raw_checkpoint, f"{path}.checkpoint_id")
        ),
        agent_rng_version=_string(payload["agent_rng_version"], f"{path}.agent_rng_version"),
    )


def _exploration_payload(value: ExplorationProvenance) -> dict[str, JsonValue]:
    return {
        "selector_version": value.selector_version,
        "temperature": value.temperature,
        "rng_version": value.rng_version,
    }


def _exploration_from_payload(value: JsonValue) -> ExplorationProvenance:
    payload = _object(value, "compact_episode.exploration")
    _exact_keys(
        payload, {"selector_version", "temperature", "rng_version"}, "compact_episode.exploration"
    )
    return ExplorationProvenance(
        selector_version=_string(
            payload["selector_version"], "compact_episode.exploration.selector_version"
        ),
        temperature=_number(payload["temperature"], "compact_episode.exploration.temperature"),
        rng_version=_string(payload["rng_version"], "compact_episode.exploration.rng_version"),
    )


def _determinization_payload(value: DeterminizationProvenance) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "sampler_version": value.sampler_version,
        "rng_version": value.rng_version,
        "count": value.count,
        "failure_policy_id": value.failure_policy_id,
        "strict": value.strict,
    }
    if value.search_descriptor_id is not None:
        payload["search_descriptor_id"] = value.search_descriptor_id
    return payload


def _determinization_from_payload(value: JsonValue) -> DeterminizationProvenance:
    payload = _object(value, "compact_episode.determinization")
    legacy_keys = {"sampler_version", "rng_version", "count", "failure_policy_id", "strict"}
    allowed_keys = {
        frozenset(legacy_keys),
        frozenset(legacy_keys | {"search_descriptor_id"}),
    }
    if set(payload) not in allowed_keys:
        _exact_keys(payload, legacy_keys, "compact_episode.determinization")
    raw_failure_policy = payload["failure_policy_id"]
    search_descriptor_id = (
        _string(
            payload["search_descriptor_id"],
            "compact_episode.determinization.search_descriptor_id",
        )
        if "search_descriptor_id" in payload
        else None
    )
    return DeterminizationProvenance(
        sampler_version=_string(
            payload["sampler_version"], "compact_episode.determinization.sampler_version"
        ),
        rng_version=_string(payload["rng_version"], "compact_episode.determinization.rng_version"),
        count=_integer(payload["count"], "compact_episode.determinization.count"),
        failure_policy_id=(
            None
            if raw_failure_policy is None
            else _string(raw_failure_policy, "compact_episode.determinization.failure_policy_id")
        ),
        strict=_boolean(payload["strict"], "compact_episode.determinization.strict"),
        search_descriptor_id=search_descriptor_id,
    )


def _schema_versions(episode: CompactEpisode) -> tuple[tuple[str, int, int], ...]:
    return (
        ("state schema", episode.state_schema_version, STATE_SCHEMA_VERSION),
        ("action schema", episode.action_schema_version, ACTION_SCHEMA_VERSION),
        ("decision schema", episode.decision_schema_version, DECISION_SCHEMA_VERSION),
        ("observation schema", episode.observation_schema_version, OBSERVATION_SCHEMA_VERSION),
        ("terminal schema", episode.terminal_schema_version, TERMINAL_SCHEMA_VERSION),
    )


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CompactReplaySchemaError(f"{path} contains a non-finite number")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise CompactReplaySchemaError(f"{path} contains non-canonical negative zero")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, f"{path}[]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CompactReplaySchemaError(f"{path} contains a non-string key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise CompactReplaySchemaError(f"{path} contains non-JSON value {type(value).__name__}")


def _object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CompactReplaySchemaError(f"{path} must be an object")
    return cast(JsonObject, value)


def _array(value: JsonValue, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise CompactReplaySchemaError(f"{path} must be an array")
    return value


def _string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str):
        raise CompactReplaySchemaError(f"{path} must be a string")
    return value


def _integer(value: JsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompactReplaySchemaError(f"{path} must be an integer")
    return value


def _number(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompactReplaySchemaError(f"{path} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise CompactReplaySchemaError(f"{path} must be finite")
    return number


def _boolean(value: JsonValue, path: str) -> bool:
    if not isinstance(value, bool):
        raise CompactReplaySchemaError(f"{path} must be a boolean")
    return value


def _player_id(value: JsonValue, path: str) -> PlayerId:
    try:
        return PlayerId(_string(value, path))
    except ValueError as error:
        raise CompactReplaySchemaError(f"{path} has unknown player ID") from error


def _exact_keys(payload: JsonObject, expected: set[str], path: str) -> None:
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing or extra:
        raise CompactReplaySchemaError(
            f"{path} keys differ: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )


def _required_text(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} cannot be empty")


def _digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a tagged lower-case SHA-256 digest")


def _deterministic_gzip(payload: bytes) -> bytes:
    output = bytearray()
    # GzipFile with an empty filename emits the fixed 1f8b08000000000002ff header at level 9.
    from io import BytesIO

    stream = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    output.extend(stream.getvalue())
    expected_header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    if not bytes(output).startswith(expected_header):  # pragma: no cover - platform defense
        raise CompactReplayShardError("runtime did not produce the required fixed gzip header")
    return bytes(output)


def _read_fixed_gzip(compressed: bytes) -> bytes:
    expected_header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    if not compressed.startswith(expected_header):
        raise CompactReplayShardError("compact replay gzip header is not the required fixed header")
    try:
        decoder = zlib.decompressobj(wbits=31)
        raw = decoder.decompress(compressed) + decoder.flush()
    except zlib.error as error:
        raise CompactReplayShardError(
            f"invalid or truncated compact replay gzip: {error}"
        ) from error
    if not decoder.eof:
        raise CompactReplayShardError("truncated compact replay gzip member")
    if decoder.unused_data:
        raise CompactReplayShardError("compact replay shard must contain exactly one gzip member")
    return raw


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise CompactReplayShardError(
            f"could not atomically seal compact replay shard {path}: {error}"
        ) from error
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


_EPISODE_KEYS = {
    "format",
    "schema_version",
    "episode_id",
    "engine_version",
    "rules_version",
    "information_policy_version",
    "state_schema_version",
    "action_schema_version",
    "decision_schema_version",
    "observation_schema_version",
    "terminal_schema_version",
    "card_data_fingerprint",
    "effects_fingerprint",
    "setup",
    "generation",
    "seat_mapping",
    "exploration",
    "determinization",
    "actions",
    "transition_count",
    "terminal_result",
    "final_state_hash",
    "producer_run_id",
    "resolved_config_digest",
}
