"""Verified compact-replay extraction and deterministic NumPy dataset materialization.

Compact episodes are the durable source of truth.  This module replays each episode exactly once,
uses only player-safe :class:`~innovation_ai.harness.policy.ValuePosition` objects for features,
and writes disposable deterministic ``.npz`` caches.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Collection, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
from numpy.lib import format as numpy_format
from numpy.typing import NDArray

from innovation_ai.harness.policy import (
    ValuePosition,
    build_afterstate_value_position,
)
from innovation_ai.innovation.actions import ActionKind, Decision, DecisionKind
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.invariants import InvariantViolation, assert_state_properties
from innovation_ai.innovation.logs import ReplayOutcome
from innovation_ai.innovation.protocol import InnovationEngineError
from innovation_ai.innovation.replay import DefaultReplayAdapter, ReplayAdapter
from innovation_ai.innovation.state import TerminalResult, state_hash
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.compact_replay import (
    CompactEpisode,
    CompactReplayDivergenceError,
    CompactReplayError,
    JsonValue,
    canonical_json,
    check_compact_episode_compatibility,
    read_compact_replay_shard,
    setup_provenance_digest,
    sha256_digest,
)
from innovation_ai.training.encoding import FlatObservationEncoder

DATASET_FORMAT = "innovation-ai-value-dataset"
DATASET_SCHEMA_VERSION = 1
EXTRACTION_POLICY_FORMAT = "innovation-ai-value-extraction-policy"
EXTRACTION_POLICY_SCHEMA_VERSION = 1
DEFAULT_SPLIT_SALT = "innovation-ai-value-dataset-split-v1"


class DatasetError(ValueError):
    """Base class for dataset extraction and materialization failures."""


class DatasetSchemaError(DatasetError):
    """A dataset policy or manifest is not an exact supported schema."""


class DatasetMaterializationError(DatasetError):
    """A source shard, output shard, or resumable output is inconsistent."""


class DatasetSplit(StrEnum):
    """The two deterministic dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    """Versioned rules selecting replay post-action examples.

    The default deliberately excludes setup afterstates and terminal states while retaining both
    paid-turn and effect-choice afterstates.  Future policy changes must use a new schema version
    or a separately named policy value in a dataset manifest.
    """

    include_starting_meld_afterstates: bool = False
    include_terminal_positions: bool = False
    include_turn_action_afterstates: bool = True
    include_effect_choice_afterstates: bool = True
    format: str = EXTRACTION_POLICY_FORMAT
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != EXTRACTION_POLICY_FORMAT:
            raise ValueError(f"unsupported extraction policy format {self.format!r}")
        if self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported extraction policy schema {self.schema_version}")


DEFAULT_EXTRACTION_POLICY = ExtractionPolicy()


@dataclass(frozen=True, slots=True)
class ValuePositionExample:
    """One terminal-labeled, player-safe afterstate reconstructed from a compact replay."""

    episode_id: str
    setup_provenance_digest: str
    action_sequence: int
    action_kind: ActionKind
    decision_kind: DecisionKind
    viewer: PlayerId
    position: ValuePosition
    target: float

    def __post_init__(self) -> None:
        if not self.episode_id or self.action_sequence < 1:
            raise ValueError("invalid extracted example identity")
        _digest(self.setup_provenance_digest, "setup provenance digest")
        if self.position.viewer is not self.viewer:
            raise ValueError("extracted example viewer differs from its position")
        if self.position.position_kind.value != "afterstate":
            raise ValueError("extracted examples must be afterstates")
        if self.target not in (0.0, 0.5, 1.0):
            raise ValueError("terminal target must be 0, 0.5, or 1")


@dataclass(frozen=True, slots=True)
class DatasetSourceShard:
    """Immutable source compact-shard identity captured by a materialized dataset."""

    shard_id: str
    sha256: str
    episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.shard_id, "source shard ID")
        _digest(self.sha256, "source shard hash")
        _episode_ids(self.episode_ids, "source shard episode IDs")


@dataclass(frozen=True, slots=True)
class DatasetSplitMember:
    """One episode's grouped train/validation assignment."""

    episode_id: str
    setup_provenance_digest: str
    split: DatasetSplit

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode ID")
        _digest(self.setup_provenance_digest, "setup provenance digest")


@dataclass(frozen=True, slots=True)
class DatasetShard:
    """One deterministic materialized NPZ shard and its preassigned episode membership."""

    shard_id: str
    split: DatasetSplit
    episode_ids: tuple[str, ...]
    sha256: str
    example_count: int

    def __post_init__(self) -> None:
        _identifier(self.shard_id, "dataset shard ID")
        _episode_ids(self.episode_ids, "dataset shard episode IDs")
        _digest(self.sha256, "dataset shard hash")
        if self.example_count < 0:
            raise ValueError("dataset shard example count cannot be negative")


@dataclass(frozen=True, slots=True)
class DatasetCounts:
    """Small auditable totals duplicated in the manifest for quick validation."""

    episode_count: int
    example_count: int
    train_episode_count: int
    validation_episode_count: int
    train_example_count: int
    validation_example_count: int

    def __post_init__(self) -> None:
        if (
            min(
                self.episode_count,
                self.example_count,
                self.train_episode_count,
                self.validation_episode_count,
                self.train_example_count,
                self.validation_example_count,
            )
            < 0
        ):
            raise ValueError("dataset counts cannot be negative")
        if self.episode_count != self.train_episode_count + self.validation_episode_count:
            raise ValueError("dataset episode split counts do not add up")
        if self.example_count != self.train_example_count + self.validation_example_count:
            raise ValueError("dataset example split counts do not add up")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Strict, versioned provenance for disposable encoded dataset shards."""

    encoder_fingerprint: str
    encoder_version: str
    extraction_policy: ExtractionPolicy
    split_salt: str
    validation_fraction: float
    source_shards: tuple[DatasetSourceShard, ...]
    split_membership: tuple[DatasetSplitMember, ...]
    shards: tuple[DatasetShard, ...]
    counts: DatasetCounts
    format: str = DATASET_FORMAT
    schema_version: int = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DATASET_FORMAT:
            raise ValueError(f"unsupported dataset format {self.format!r}")
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset schema {self.schema_version}")
        _digest(self.encoder_fingerprint, "encoder fingerprint")
        if not self.encoder_version or not self.split_salt:
            raise ValueError("dataset encoder version and split salt cannot be empty")
        if (
            not math.isfinite(self.validation_fraction)
            or not 0.0 <= self.validation_fraction <= 1.0
        ):
            raise ValueError("validation fraction must be finite and in [0, 1]")
        if not self.source_shards:
            raise ValueError("dataset manifest must have source shards")
        source_ids = tuple(item.shard_id for item in self.source_shards)
        if tuple(sorted(source_ids)) != source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("source shards must have unique canonical IDs")
        members = tuple(item.episode_id for item in self.split_membership)
        if tuple(sorted(members)) != members or len(set(members)) != len(members):
            raise ValueError("split membership must have unique canonical episode IDs")
        source_episode_ids = tuple(
            episode_id for source in self.source_shards for episode_id in source.episode_ids
        )
        if tuple(sorted(source_episode_ids)) != members:
            raise ValueError("split membership must exactly cover source episode IDs")
        memberships = {item.episode_id: item for item in self.split_membership}
        digest_splits: dict[str, DatasetSplit] = {}
        for member in self.split_membership:
            previous = digest_splits.setdefault(member.setup_provenance_digest, member.split)
            if previous is not member.split:
                raise ValueError("identical setup provenance crosses the dataset split")
        shard_ids = tuple(item.shard_id for item in self.shards)
        if tuple(sorted(shard_ids)) != shard_ids or len(set(shard_ids)) != len(shard_ids):
            raise ValueError("dataset shards must have unique canonical IDs")
        assigned = tuple(episode_id for shard in self.shards for episode_id in shard.episode_ids)
        if tuple(sorted(assigned)) != members:
            raise ValueError("dataset shards must exactly cover source episode IDs")
        for shard in self.shards:
            if any(
                memberships[episode_id].split is not shard.split for episode_id in shard.episode_ids
            ):
                raise ValueError("dataset shard mixes split membership")
        expected = DatasetCounts(
            episode_count=len(members),
            example_count=sum(shard.example_count for shard in self.shards),
            train_episode_count=sum(
                item.split is DatasetSplit.TRAIN for item in self.split_membership
            ),
            validation_episode_count=sum(
                item.split is DatasetSplit.VALIDATION for item in self.split_membership
            ),
            train_example_count=sum(
                shard.example_count for shard in self.shards if shard.split is DatasetSplit.TRAIN
            ),
            validation_example_count=sum(
                shard.example_count
                for shard in self.shards
                if shard.split is DatasetSplit.VALIDATION
            ),
        )
        if self.counts != expected:
            raise ValueError("dataset manifest counts differ from its membership or shards")


def extraction_policy_payload(policy: ExtractionPolicy) -> dict[str, object]:
    """Return the canonical strict payload for an extraction policy."""

    return {
        "format": policy.format,
        "schema_version": policy.schema_version,
        "include_starting_meld_afterstates": policy.include_starting_meld_afterstates,
        "include_terminal_positions": policy.include_terminal_positions,
        "include_turn_action_afterstates": policy.include_turn_action_afterstates,
        "include_effect_choice_afterstates": policy.include_effect_choice_afterstates,
    }


def extraction_policy_from_payload(payload: object) -> ExtractionPolicy:
    """Decode an exact extraction-policy payload."""

    value = _object(payload, "extraction policy")
    _exact_keys(value, _EXTRACTION_POLICY_KEYS, "extraction policy")
    try:
        return ExtractionPolicy(
            include_starting_meld_afterstates=_boolean(
                value["include_starting_meld_afterstates"], "starting-meld inclusion"
            ),
            include_terminal_positions=_boolean(
                value["include_terminal_positions"], "terminal inclusion"
            ),
            include_turn_action_afterstates=_boolean(
                value["include_turn_action_afterstates"], "turn-action inclusion"
            ),
            include_effect_choice_afterstates=_boolean(
                value["include_effect_choice_afterstates"], "effect-choice inclusion"
            ),
            format=_string(value["format"], "extraction policy format"),
            schema_version=_integer(value["schema_version"], "extraction policy schema"),
        )
    except ValueError as error:
        raise DatasetSchemaError(f"invalid extraction policy: {error}") from error


def dataset_manifest_payload(manifest: DatasetManifest) -> dict[str, object]:
    """Return the complete canonical JSON-compatible dataset manifest payload."""

    return {
        "format": manifest.format,
        "schema_version": manifest.schema_version,
        "encoder_fingerprint": manifest.encoder_fingerprint,
        "encoder_version": manifest.encoder_version,
        "extraction_policy": extraction_policy_payload(manifest.extraction_policy),
        "split_salt": manifest.split_salt,
        "validation_fraction": manifest.validation_fraction,
        "source_shards": [
            {
                "shard_id": item.shard_id,
                "sha256": item.sha256,
                "episode_ids": list(item.episode_ids),
            }
            for item in manifest.source_shards
        ],
        "split_membership": [
            {
                "episode_id": item.episode_id,
                "setup_provenance_digest": item.setup_provenance_digest,
                "split": item.split.value,
            }
            for item in manifest.split_membership
        ],
        "shards": [
            {
                "shard_id": item.shard_id,
                "split": item.split.value,
                "episode_ids": list(item.episode_ids),
                "sha256": item.sha256,
                "example_count": item.example_count,
            }
            for item in manifest.shards
        ],
        "counts": {
            "episode_count": manifest.counts.episode_count,
            "example_count": manifest.counts.example_count,
            "train_episode_count": manifest.counts.train_episode_count,
            "validation_episode_count": manifest.counts.validation_episode_count,
            "train_example_count": manifest.counts.train_example_count,
            "validation_example_count": manifest.counts.validation_example_count,
        },
    }


def dumps_dataset_manifest(manifest: DatasetManifest) -> str:
    """Serialize a deterministic, strict dataset manifest."""

    return canonical_json(cast(JsonValue, dataset_manifest_payload(manifest)))


def loads_dataset_manifest(text: str) -> DatasetManifest:
    """Parse a canonical strict dataset manifest."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise DatasetSchemaError(f"invalid dataset manifest JSON: {error}") from error
    if canonical_json(cast(JsonValue, payload)) != text:
        raise DatasetSchemaError("dataset manifest is not canonical JSON")
    return dataset_manifest_from_payload(payload)


def dataset_manifest_from_payload(payload: object) -> DatasetManifest:
    """Decode a strict dataset manifest payload, rejecting missing and extra fields."""

    value = _object(payload, "dataset manifest")
    _exact_keys(value, _DATASET_MANIFEST_KEYS, "dataset manifest")
    try:
        source_shards = tuple(
            _source_shard_from_payload(item) for item in _array(value["source_shards"])
        )
        members = tuple(
            _split_member_from_payload(item) for item in _array(value["split_membership"])
        )
        shards = tuple(_dataset_shard_from_payload(item) for item in _array(value["shards"]))
        counts = _counts_from_payload(value["counts"])
        return DatasetManifest(
            encoder_fingerprint=_string(value["encoder_fingerprint"], "encoder fingerprint"),
            encoder_version=_string(value["encoder_version"], "encoder version"),
            extraction_policy=extraction_policy_from_payload(value["extraction_policy"]),
            split_salt=_string(value["split_salt"], "split salt"),
            validation_fraction=_number(value["validation_fraction"], "validation fraction"),
            source_shards=source_shards,
            split_membership=members,
            shards=shards,
            counts=counts,
            format=_string(value["format"], "dataset format"),
            schema_version=_integer(value["schema_version"], "dataset schema"),
        )
    except ValueError as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetSchemaError(f"invalid dataset manifest: {error}") from error


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> str:
    """Atomically write one canonical manifest with a trailing newline and return its digest."""

    encoded = (dumps_dataset_manifest(manifest) + "\n").encode("ascii")
    _atomic_write(path, encoded)
    return sha256_digest(encoded)


def read_dataset_manifest(path: Path) -> DatasetManifest:
    """Read one canonical dataset manifest written by :func:`write_dataset_manifest`."""

    try:
        text = path.read_text(encoding="ascii")
    except OSError as error:
        raise DatasetMaterializationError(f"could not read dataset manifest: {error}") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise DatasetSchemaError("dataset manifest must have exactly one trailing newline")
    return loads_dataset_manifest(text[:-1])


def terminal_utility(result: TerminalResult, viewer: PlayerId) -> float:
    """Return the terminal utility in ``viewer``'s perspective: win/draw/loss = 1/.5/0."""

    if result.is_draw:
        return 0.5
    return 1.0 if viewer in result.winners else 0.0


def extract_value_position_examples(
    episode: CompactEpisode,
    registry: CardRegistry | None = None,
    *,
    policy: ExtractionPolicy = DEFAULT_EXTRACTION_POLICY,
    adapter: ReplayAdapter | None = None,
) -> tuple[ValuePositionExample, ...]:
    """Replay one compact episode once, verify it, and return selected labeled afterstates.

    Each feature position is built *after* the recorded action from the action's original chooser
    and a freshly reconstructed next decision boundary.  The only future-derived datum attached
    is the terminal target.
    """

    registry = registry or load_card_registry()
    selected_adapter = adapter or DefaultReplayAdapter()
    check_compact_episode_compatibility(episode, registry)
    try:
        state = selected_adapter.initial_state(episode.setup, registry)
        assert_state_properties(state, registry)
    except (ValueError, InvariantViolation) as error:
        raise CompactReplayDivergenceError(
            f"initial setup reconstruction failed: {error}"
        ) from error

    pending: list[tuple[int, ActionKind, DecisionKind, PlayerId, ValuePosition]] = []
    for sequence, action in enumerate(episode.actions, start=1):
        decisions = selected_adapter.decisions(state, registry)
        decision = _decision_for_action(decisions, action.decision_id, sequence)
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

        is_terminal = selected_adapter.outcome(state) is ReplayOutcome.TERMINAL
        if _include_afterstate(policy, decision.kind, is_terminal):
            next_decision = _next_boundary(
                selected_adapter.decisions(state, registry), decision.chooser
            )
            position = build_afterstate_value_position(
                state,
                decision.chooser,
                next_decision,
                registry,
            )
            pending.append((sequence, action.kind, decision.kind, decision.chooser, position))

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
    terminal = state.terminal_result
    if terminal is None:  # pragma: no cover - adapter contract defense
        raise CompactReplayDivergenceError("terminal replay state lacks a terminal result")
    digest = setup_provenance_digest(episode.setup)
    return tuple(
        ValuePositionExample(
            episode_id=episode.episode_id,
            setup_provenance_digest=digest,
            action_sequence=sequence,
            action_kind=action_kind,
            decision_kind=decision_kind,
            viewer=viewer,
            position=position,
            target=terminal_utility(terminal, viewer),
        )
        for sequence, action_kind, decision_kind, viewer, position in pending
    )


# A concise alias makes the extraction operation easy to discover from an interactive session.
extract_examples = extract_value_position_examples


def split_for_setup_provenance(
    provenance_digest: str,
    *,
    validation_fraction: float,
    split_salt: str = DEFAULT_SPLIT_SALT,
) -> DatasetSplit:
    """Assign every equal full setup provenance digest to the same deterministic split."""

    _digest(provenance_digest, "setup provenance digest")
    if (
        not split_salt
        or not math.isfinite(validation_fraction)
        or not 0.0 <= validation_fraction <= 1.0
    ):
        raise ValueError("split salt and finite validation fraction in [0, 1] are required")
    if validation_fraction == 0.0:
        return DatasetSplit.TRAIN
    if validation_fraction == 1.0:
        return DatasetSplit.VALIDATION
    # Integer arithmetic avoids platform-dependent float comparison after the input is validated.
    value = int(hashlib.sha256(f"{split_salt}:{provenance_digest}".encode("ascii")).hexdigest(), 16)
    threshold = int(validation_fraction * (1 << 256))
    return DatasetSplit.VALIDATION if value < threshold else DatasetSplit.TRAIN


def materialize_dataset(
    source_paths: Sequence[Path],
    output_directory: Path,
    *,
    encoder: FlatObservationEncoder | None = None,
    extraction_policy: ExtractionPolicy = DEFAULT_EXTRACTION_POLICY,
    validation_fraction: float = 0.2,
    split_salt: str = DEFAULT_SPLIT_SALT,
    episodes_per_shard: int = 256,
    registry: CardRegistry | None = None,
    adapter: ReplayAdapter | None = None,
) -> DatasetManifest:
    """Verify compact source shards and build deterministic, resumable encoded NPZ shards.

    Existing output shards are accepted only if their exact bytes already match the deterministic
    materialization.  This makes interrupted runs resumable while rejecting stale/corrupt output.
    """

    if not source_paths:
        raise ValueError("at least one compact replay source shard is required")
    if episodes_per_shard < 1:
        raise ValueError("episodes per shard must be positive")
    registry = registry or load_card_registry()
    encoder = encoder or FlatObservationEncoder(registry)
    source_records = _read_source_records(source_paths, registry)
    episodes = tuple(episode for record in source_records for episode in record.episodes)
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise DatasetMaterializationError("compact source shards contain duplicate episode IDs")
    episodes = tuple(sorted(episodes, key=lambda item: item.episode_id))

    examples_by_episode = {
        episode.episode_id: extract_value_position_examples(
            episode,
            registry,
            policy=extraction_policy,
            adapter=adapter,
        )
        for episode in episodes
    }
    memberships = tuple(
        DatasetSplitMember(
            episode_id=episode.episode_id,
            setup_provenance_digest=setup_provenance_digest(episode.setup),
            split=split_for_setup_provenance(
                setup_provenance_digest(episode.setup),
                validation_fraction=validation_fraction,
                split_salt=split_salt,
            ),
        )
        for episode in episodes
    )
    membership_by_id = {item.episode_id: item for item in memberships}
    source_shards = tuple(
        DatasetSourceShard(
            record.shard_id, record.sha256, tuple(item.episode_id for item in record.episodes)
        )
        for record in source_records
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    shards: list[DatasetShard] = []
    for split in DatasetSplit:
        assigned = tuple(
            episode for episode in episodes if membership_by_id[episode.episode_id].split is split
        )
        for index, chunk in enumerate(_chunks(assigned, episodes_per_shard)):
            shard_id = f"{split.value}-{index:05d}"
            examples = tuple(
                example for episode in chunk for example in examples_by_episode[episode.episode_id]
            )
            encoded = _encode_examples(examples, encoder)
            data = _deterministic_npz(encoded)
            path = output_directory / f"{shard_id}.npz"
            _write_resumable(path, data)
            shards.append(
                DatasetShard(
                    shard_id=shard_id,
                    split=split,
                    episode_ids=tuple(episode.episode_id for episode in chunk),
                    sha256=sha256_digest(data),
                    example_count=len(examples),
                )
            )

    shards.sort(key=lambda item: item.shard_id)
    counts = DatasetCounts(
        episode_count=len(episodes),
        example_count=sum(item.example_count for item in shards),
        train_episode_count=sum(item.split is DatasetSplit.TRAIN for item in memberships),
        validation_episode_count=sum(item.split is DatasetSplit.VALIDATION for item in memberships),
        train_example_count=sum(
            item.example_count for item in shards if item.split is DatasetSplit.TRAIN
        ),
        validation_example_count=sum(
            item.example_count for item in shards if item.split is DatasetSplit.VALIDATION
        ),
    )
    manifest = DatasetManifest(
        encoder_fingerprint=encoder.manifest.layout_fingerprint,
        encoder_version=encoder.manifest.encoder_version,
        extraction_policy=extraction_policy,
        split_salt=split_salt,
        validation_fraction=validation_fraction,
        source_shards=source_shards,
        split_membership=memberships,
        shards=tuple(shards),
        counts=counts,
    )
    _write_resumable(
        output_directory / "manifest.json",
        (dumps_dataset_manifest(manifest) + "\n").encode("ascii"),
    )
    return manifest


def load_dataset_shard(path: Path) -> dict[str, NDArray[np.generic]]:
    """Load and validate the small stable NPZ array contract from a materialized shard."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_ARRAY_NAMES:
                raise DatasetMaterializationError("dataset NPZ fields differ from schema")
            arrays = {name: archive[name] for name in _NPZ_ARRAY_NAMES}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, DatasetMaterializationError):
            raise
        raise DatasetMaterializationError(f"could not load dataset shard: {error}") from error
    _validate_npz_arrays(arrays)
    return arrays


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    shard_id: str
    sha256: str
    episodes: tuple[CompactEpisode, ...]


def _read_source_records(
    source_paths: Sequence[Path], registry: CardRegistry
) -> tuple[_SourceRecord, ...]:
    paths = tuple(sorted((Path(path) for path in source_paths), key=lambda path: str(path)))
    records: list[_SourceRecord] = []
    for path in paths:
        shard_id = _source_shard_id(path)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise DatasetMaterializationError(
                f"could not read compact source shard {path}: {error}"
            ) from error
        try:
            episodes = read_compact_replay_shard(path, verify=False, registry=registry)
        except CompactReplayError as error:
            raise DatasetMaterializationError(
                f"could not parse compact source shard {path}: {error}"
            ) from error
        records.append(_SourceRecord(shard_id, sha256_digest(raw), episodes))
    records.sort(key=lambda item: item.shard_id)
    if len({item.shard_id for item in records}) != len(records):
        raise DatasetMaterializationError("compact source shards have duplicate derived shard IDs")
    return tuple(records)


def _source_shard_id(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".json.gz", ".gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _include_afterstate(
    policy: ExtractionPolicy, decision_kind: DecisionKind, is_terminal: bool
) -> bool:
    if is_terminal and not policy.include_terminal_positions:
        return False
    return {
        DecisionKind.STARTING_MELD: policy.include_starting_meld_afterstates,
        DecisionKind.TURN_ACTION: policy.include_turn_action_afterstates,
        DecisionKind.EFFECT_CHOICE: policy.include_effect_choice_afterstates,
    }[decision_kind]


def _decision_for_action(
    decisions: tuple[Decision, ...], action_decision_id: int, sequence: int
) -> Decision:
    matches = tuple(
        decision for decision in decisions if decision.decision_id == action_decision_id
    )
    if len(matches) != 1:
        raise CompactReplayDivergenceError(
            f"transition {sequence}: decision {action_decision_id} is not pending"
        )
    return matches[0]


def _next_boundary(decisions: tuple[Decision, ...], original_viewer: PlayerId) -> Decision | None:
    """Choose the semantic next boundary deterministically without reusing a stale observation."""

    if len(decisions) < 2:
        return decisions[0] if decisions else None
    own = tuple(decision for decision in decisions if decision.chooser is original_viewer)
    return own[0] if len(own) == 1 else decisions[0]


def _encode_examples(
    examples: tuple[ValuePositionExample, ...], encoder: FlatObservationEncoder
) -> dict[str, NDArray[np.generic]]:
    dimension = encoder.manifest.input_dimension
    features: NDArray[np.float32]
    if examples:
        features = encoder.encode_batch(tuple(example.position for example in examples))
    else:
        features = np.empty((0, dimension), dtype=np.float32)
    max_id = max((len(example.episode_id) for example in examples), default=1)
    arrays: dict[str, NDArray[np.generic]] = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "targets": np.asarray([example.target for example in examples], dtype=np.float32),
        "episode_ids": np.asarray(
            [example.episode_id for example in examples], dtype=f"<U{max_id}"
        ),
        "viewers": np.asarray(
            [_player_code(example.viewer) for example in examples], dtype=np.uint8
        ),
        "action_kinds": np.asarray(
            [_enum_code(ActionKind, example.action_kind) for example in examples], dtype=np.uint8
        ),
        "decision_kinds": np.asarray(
            [_enum_code(DecisionKind, example.decision_kind) for example in examples],
            dtype=np.uint8,
        ),
        "action_sequences": np.asarray(
            [example.action_sequence for example in examples], dtype=np.uint32
        ),
    }
    _validate_npz_arrays(arrays, dimension=dimension)
    return arrays


def _deterministic_npz(arrays: dict[str, NDArray[np.generic]]) -> bytes:
    """Build stable stored-ZIP NPZ bytes (NumPy's convenience writer timestamps ZIP entries)."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
    ) as archive:
        for name in sorted(_NPZ_ARRAY_NAMES):
            item = io.BytesIO()
            numpy_format.write_array(item, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, item.getvalue())
    return buffer.getvalue()


def _validate_npz_arrays(
    arrays: dict[str, NDArray[np.generic]], *, dimension: int | None = None
) -> None:
    features = arrays["features"]
    targets = arrays["targets"]
    if features.dtype != np.dtype(np.float32) or features.ndim != 2:
        raise DatasetMaterializationError("dataset features must be rank-2 float32")
    if dimension is not None and features.shape[1] != dimension:
        raise DatasetMaterializationError("dataset feature dimension differs from encoder")
    count = features.shape[0]
    expected: dict[str, np.dtype[np.generic]] = {
        "targets": np.dtype(np.float32),
        "episode_ids": arrays["episode_ids"].dtype,
        "viewers": np.dtype(np.uint8),
        "action_kinds": np.dtype(np.uint8),
        "decision_kinds": np.dtype(np.uint8),
        "action_sequences": np.dtype(np.uint32),
    }
    if targets.ndim != 1 or targets.dtype != expected["targets"]:
        raise DatasetMaterializationError("dataset targets must be rank-1 float32")
    if arrays["episode_ids"].dtype.kind != "U" or arrays["episode_ids"].ndim != 1:
        raise DatasetMaterializationError("dataset episode IDs must be rank-1 unicode")
    for name in ("viewers", "action_kinds", "decision_kinds", "action_sequences"):
        if arrays[name].dtype != expected[name] or arrays[name].ndim != 1:
            raise DatasetMaterializationError(f"dataset {name} has wrong dtype or rank")
    if any(arrays[name].shape[0] != count for name in _NPZ_ARRAY_NAMES if name != "features"):
        raise DatasetMaterializationError("dataset array lengths differ")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise DatasetMaterializationError("dataset arrays contain non-finite values")
    if not np.isin(targets, np.asarray((0.0, 0.5, 1.0), dtype=np.float32)).all():
        raise DatasetMaterializationError("dataset targets are not terminal utilities")


def _write_resumable(path: Path, data: bytes) -> None:
    """Accept an exact completed output, otherwise atomically publish the deterministic bytes."""

    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise DatasetMaterializationError(
                f"could not read existing output {path}: {error}"
            ) from error
        if existing != data:
            raise DatasetMaterializationError(
                f"existing output {path} differs from deterministic materialization"
            )
        return
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _chunks[T](items: tuple[T, ...], size: int) -> Iterable[tuple[T, ...]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _player_code(player: PlayerId) -> int:
    return tuple(PlayerId).index(player)


def _enum_code[T: StrEnum](enum_type: type[T], value: T) -> int:
    return tuple(enum_type).index(value)


def _source_shard_from_payload(payload: object) -> DatasetSourceShard:
    value = _object(payload, "source shard")
    _exact_keys(value, {"shard_id", "sha256", "episode_ids"}, "source shard")
    return DatasetSourceShard(
        _string(value["shard_id"], "source shard ID"),
        _string(value["sha256"], "source shard hash"),
        _strings(value["episode_ids"], "source shard episode IDs"),
    )


def _split_member_from_payload(payload: object) -> DatasetSplitMember:
    value = _object(payload, "split member")
    _exact_keys(value, {"episode_id", "setup_provenance_digest", "split"}, "split member")
    try:
        split = DatasetSplit(_string(value["split"], "split member split"))
    except ValueError as error:
        raise DatasetSchemaError("invalid split member split") from error
    return DatasetSplitMember(
        _string(value["episode_id"], "split member episode ID"),
        _string(value["setup_provenance_digest"], "split member setup digest"),
        split,
    )


def _dataset_shard_from_payload(payload: object) -> DatasetShard:
    value = _object(payload, "dataset shard")
    _exact_keys(
        value, {"shard_id", "split", "episode_ids", "sha256", "example_count"}, "dataset shard"
    )
    try:
        split = DatasetSplit(_string(value["split"], "dataset shard split"))
    except ValueError as error:
        raise DatasetSchemaError("invalid dataset shard split") from error
    return DatasetShard(
        _string(value["shard_id"], "dataset shard ID"),
        split,
        _strings(value["episode_ids"], "dataset shard episode IDs"),
        _string(value["sha256"], "dataset shard hash"),
        _integer(value["example_count"], "dataset shard example count"),
    )


def _counts_from_payload(payload: object) -> DatasetCounts:
    value = _object(payload, "dataset counts")
    _exact_keys(value, _COUNT_KEYS, "dataset counts")
    return DatasetCounts(*(_integer(value[name], f"dataset counts {name}") for name in _COUNT_KEYS))


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetSchemaError(f"{name} must be an object")
    return value


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise DatasetSchemaError("dataset array field must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise DatasetSchemaError(f"{name} must be a string")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    items = _array(value)
    if not all(isinstance(item, str) for item in items):
        raise DatasetSchemaError(f"{name} must be a string array")
    return tuple(cast(str, item) for item in items)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetSchemaError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetSchemaError(f"{name} must be a finite number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetSchemaError(f"{name} must be a boolean")
    return value


def _exact_keys(payload: dict[str, object], expected: Collection[str], name: str) -> None:
    if set(payload) != set(expected):
        raise DatasetSchemaError(f"{name} fields differ from schema")


def _identifier(value: str, name: str) -> None:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError(f"invalid {name}")


def _episode_ids(values: tuple[str, ...], name: str) -> None:
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty unique canonical order")
    for value in values:
        _identifier(value, name)


def _digest(value: str, name: str) -> None:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"invalid {name}")


_EXTRACTION_POLICY_KEYS = {
    "format",
    "schema_version",
    "include_starting_meld_afterstates",
    "include_terminal_positions",
    "include_turn_action_afterstates",
    "include_effect_choice_afterstates",
}
_DATASET_MANIFEST_KEYS = {
    "format",
    "schema_version",
    "encoder_fingerprint",
    "encoder_version",
    "extraction_policy",
    "split_salt",
    "validation_fraction",
    "source_shards",
    "split_membership",
    "shards",
    "counts",
}
_COUNT_KEYS = (
    "episode_count",
    "example_count",
    "train_episode_count",
    "validation_episode_count",
    "train_example_count",
    "validation_example_count",
)
_NPZ_ARRAY_NAMES = {
    "features",
    "targets",
    "episode_ids",
    "viewers",
    "action_kinds",
    "decision_kinds",
    "action_sequences",
}
