"""Immutable, digest-verified value-model checkpoint bundles and policy identities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from innovation_ai.innovation.actions import ACTION_SCHEMA_VERSION, DECISION_SCHEMA_VERSION
from innovation_ai.innovation.effects.model import EFFECT_RUNTIME_SCHEMA_VERSION
from innovation_ai.innovation.effects.registry import effects_fingerprint
from innovation_ai.innovation.logs import ENGINE_VERSION
from innovation_ai.innovation.observations import OBSERVATION_SCHEMA_VERSION
from innovation_ai.innovation.state import (
    INFORMATION_POLICY_VERSION,
    PUBLIC_COVERED_INFORMATION_POLICY_VERSION,
    RULES_VERSION,
    STATE_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
)
from innovation_ai.search.contracts import PRODUCTION_SEARCH_DESCRIPTOR
from innovation_ai.training.encoding import EncoderManifest, build_encoder_manifest
from innovation_ai.training.model import (
    VALUE_NETWORK_ARCHITECTURE,
    VALUE_NETWORK_HIDDEN_DIMENSION,
    ValueNetwork,
)

if TYPE_CHECKING:
    from innovation_ai.training.inference import CpuBatchValueEvaluator

CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION = 1
POLICY_DESCRIPTOR_SCHEMA_VERSION = 2

DEFAULT_AFTERSTATE_BOUNDARY_SEMANTICS_VERSION = "immediate-one-transition-v1"
DEFAULT_INFORMATION_SET_SAMPLER_VERSION = "information-set-sampler-v1"
DEFAULT_SAMPLER_RNG_VERSION = "sha256-counter-v1"
LEGACY_DEFAULT_FALLBACK_AGENT = "simple-heuristic"
DEFAULT_FALLBACK_AGENT = "sampled-minimax-heuristic"
DEFAULT_FALLBACK_AGENT_VERSION = "v1"
DEFAULT_LEARNED_TURN_ACTION_POLICY = "sampled-afterstate-value-v1"
DEFAULT_SEARCH_CONTINUATION_POLICY = "hand-engineered-minimax-both-players-v1"
DEFAULT_SELECTOR_VERSION = "temperature-softmax-v1"
DEFAULT_SELECTOR_RNG_VERSION = "sha256-domain-separated-v1"


class CheckpointError(ValueError):
    """Base error for invalid or unavailable checkpoint bundles."""


class CheckpointIntegrityError(CheckpointError):
    """A checkpoint file does not match its manifest or safe state-dict contract."""


class CheckpointCompatibilityError(CheckpointError):
    """A valid bundle is incompatible with the installed model/engine/encoder."""


class PolicyCompatibilityError(CheckpointCompatibilityError):
    """A policy descriptor does not describe its referenced checkpoint."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value must be JSON-serializable without NaN or infinity") from error


def _canonical_json_bytes(value: object) -> bytes:
    return f"{_canonical_json(value)}\n".encode()


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointIntegrityError(f"{name} must be a non-empty string")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CheckpointIntegrityError(f"{name} must be a string")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointIntegrityError(f"{name} must be an integer >= {minimum}")
    return value


def _require_string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise CheckpointIntegrityError(f"{name} must be a non-empty-string array")
    return tuple(value)


def _require_json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CheckpointIntegrityError(f"{name} must be a JSON object")
    # This both validates recursive JSON values and rejects NaN/Infinity.
    return cast(dict[str, object], json.loads(_canonical_json(value)))


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Strict metadata for one immutable checkpoint directory.

    The ``checkpoint_id`` is derived from all metadata and file digests.  It is
    deliberately not caller-supplied, so a checkpoint cannot be renamed into a
    different identity after publication.
    """

    architecture: str
    input_dimension: int
    hidden_dimension: int
    encoder_version: str
    encoder_layout_fingerprint: str
    card_data_fingerprint: str
    rules_version: str
    information_policy_version: str
    engine_version: str
    effects_fingerprint: str
    action_schema_version: int
    decision_schema_version: int
    observation_schema_version: int
    state_schema_version: int
    terminal_schema_version: int
    effect_runtime_schema_version: int
    value_position_schema_version: int
    public_boundary_schema_version: int
    training_dataset_ids: tuple[str, ...]
    parent_checkpoint_ids: tuple[str, ...]
    generation: int
    optimizer_config_json: str
    pytorch_version: str
    creation_command: str
    model_sha256: str
    optimizer_sha256: str | None
    metrics_sha256: str
    schema_version: int = CHECKPOINT_MANIFEST_SCHEMA_VERSION
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"unsupported checkpoint manifest schema {self.schema_version}"
            )
        if self.architecture != VALUE_NETWORK_ARCHITECTURE:
            raise CheckpointCompatibilityError("unsupported value-network architecture")
        if self.input_dimension < 1:
            raise ValueError("checkpoint input dimension must be positive")
        if self.hidden_dimension != VALUE_NETWORK_HIDDEN_DIMENSION:
            raise CheckpointCompatibilityError("checkpoint hidden dimension is incompatible")
        if self.generation < 0:
            raise ValueError("checkpoint generation cannot be negative")
        for name in (
            "encoder_version",
            "encoder_layout_fingerprint",
            "card_data_fingerprint",
            "rules_version",
            "information_policy_version",
            "engine_version",
            "effects_fingerprint",
            "pytorch_version",
            "model_sha256",
            "metrics_sha256",
        ):
            _require_string(getattr(self, name), f"checkpoint manifest {name}")
        if self.optimizer_sha256 is not None:
            _require_string(self.optimizer_sha256, "checkpoint manifest optimizer_sha256")
        if not isinstance(self.creation_command, str):
            raise ValueError("checkpoint creation command must be a string")
        if len(set(self.training_dataset_ids)) != len(self.training_dataset_ids):
            raise ValueError("checkpoint training dataset IDs cannot repeat")
        if len(set(self.parent_checkpoint_ids)) != len(self.parent_checkpoint_ids):
            raise ValueError("checkpoint parent IDs cannot repeat")
        _canonical_json(json.loads(self.optimizer_config_json))
        object.__setattr__(self, "checkpoint_id", _checkpoint_id(self._identity_payload()))

    @property
    def optimizer_config(self) -> dict[str, object]:
        """Return a detached JSON-compatible optimizer configuration object."""

        value = json.loads(self.optimizer_config_json)
        if not isinstance(value, dict):  # defensive: constructor validates canonical JSON only
            raise CheckpointIntegrityError("checkpoint optimizer config is not an object")
        return cast(dict[str, object], value)

    def _identity_payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "checkpoint_id"
        }
        payload["training_dataset_ids"] = list(self.training_dataset_ids)
        payload["parent_checkpoint_ids"] = list(self.parent_checkpoint_ids)
        payload["optimizer_config"] = json.loads(self.optimizer_config_json)
        payload.pop("optimizer_config_json")
        return cast(dict[str, object], payload)

    def payload(self) -> dict[str, object]:
        """Return canonical JSON-compatible metadata including the derived ID."""

        payload = self._identity_payload()
        payload["checkpoint_id"] = self.checkpoint_id
        return payload

    def dumps(self) -> str:
        """Serialize the exact manifest in canonical JSON form."""

        return _canonical_json(self.payload())

    @classmethod
    def from_payload(cls, payload: object) -> CheckpointManifest:
        """Decode a strict manifest and verify its content-derived ID."""

        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise CheckpointIntegrityError("checkpoint manifest must be a JSON object")
        expected = {
            "architecture",
            "input_dimension",
            "hidden_dimension",
            "encoder_version",
            "encoder_layout_fingerprint",
            "card_data_fingerprint",
            "rules_version",
            "information_policy_version",
            "engine_version",
            "effects_fingerprint",
            "action_schema_version",
            "decision_schema_version",
            "observation_schema_version",
            "state_schema_version",
            "terminal_schema_version",
            "effect_runtime_schema_version",
            "value_position_schema_version",
            "public_boundary_schema_version",
            "training_dataset_ids",
            "parent_checkpoint_ids",
            "generation",
            "optimizer_config",
            "pytorch_version",
            "creation_command",
            "model_sha256",
            "optimizer_sha256",
            "metrics_sha256",
            "schema_version",
            "checkpoint_id",
        }
        if set(payload) != expected:
            raise CheckpointIntegrityError("checkpoint manifest fields differ from schema")
        optimizer_config = _require_json_object(payload["optimizer_config"], "optimizer_config")
        value = cls(
            architecture=_require_string(payload["architecture"], "architecture"),
            input_dimension=_require_int(payload["input_dimension"], "input_dimension", minimum=1),
            hidden_dimension=_require_int(
                payload["hidden_dimension"], "hidden_dimension", minimum=1
            ),
            encoder_version=_require_string(payload["encoder_version"], "encoder_version"),
            encoder_layout_fingerprint=_require_string(
                payload["encoder_layout_fingerprint"], "encoder_layout_fingerprint"
            ),
            card_data_fingerprint=_require_string(
                payload["card_data_fingerprint"], "card_data_fingerprint"
            ),
            rules_version=_require_string(payload["rules_version"], "rules_version"),
            information_policy_version=_require_string(
                payload["information_policy_version"], "information_policy_version"
            ),
            engine_version=_require_string(payload["engine_version"], "engine_version"),
            effects_fingerprint=_require_string(
                payload["effects_fingerprint"], "effects_fingerprint"
            ),
            action_schema_version=_require_int(
                payload["action_schema_version"], "action_schema_version"
            ),
            decision_schema_version=_require_int(
                payload["decision_schema_version"], "decision_schema_version"
            ),
            observation_schema_version=_require_int(
                payload["observation_schema_version"], "observation_schema_version"
            ),
            state_schema_version=_require_int(
                payload["state_schema_version"], "state_schema_version"
            ),
            terminal_schema_version=_require_int(
                payload["terminal_schema_version"], "terminal_schema_version"
            ),
            effect_runtime_schema_version=_require_int(
                payload["effect_runtime_schema_version"], "effect_runtime_schema_version"
            ),
            value_position_schema_version=_require_int(
                payload["value_position_schema_version"], "value_position_schema_version"
            ),
            public_boundary_schema_version=_require_int(
                payload["public_boundary_schema_version"], "public_boundary_schema_version"
            ),
            training_dataset_ids=_require_string_list(
                payload["training_dataset_ids"], "training_dataset_ids"
            ),
            parent_checkpoint_ids=_require_string_list(
                payload["parent_checkpoint_ids"], "parent_checkpoint_ids"
            ),
            generation=_require_int(payload["generation"], "generation"),
            optimizer_config_json=_canonical_json(optimizer_config),
            pytorch_version=_require_string(payload["pytorch_version"], "pytorch_version"),
            creation_command=_require_text(payload["creation_command"], "creation_command"),
            model_sha256=_require_string(payload["model_sha256"], "model_sha256"),
            optimizer_sha256=(
                None
                if payload["optimizer_sha256"] is None
                else _require_string(payload["optimizer_sha256"], "optimizer_sha256")
            ),
            metrics_sha256=_require_string(payload["metrics_sha256"], "metrics_sha256"),
            schema_version=_require_int(payload["schema_version"], "schema_version", minimum=1),
        )
        stored_id = _require_string(payload["checkpoint_id"], "checkpoint_id")
        if stored_id != value.checkpoint_id:
            raise CheckpointIntegrityError("checkpoint manifest content-derived ID is invalid")
        return value


def _checkpoint_id(payload: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """Immutable complete learned-policy identity, distinct from checkpoint weights."""

    checkpoint_id: str
    encoder_layout_fingerprint: str
    card_data_fingerprint: str
    effects_fingerprint: str
    engine_version: str = ENGINE_VERSION
    rules_version: str = RULES_VERSION
    information_policy_version: str = INFORMATION_POLICY_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    decision_schema_version: int = DECISION_SCHEMA_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    value_position_schema_version: int = 1
    public_boundary_schema_version: int = 1
    afterstate_boundary_semantics_version: str = DEFAULT_AFTERSTATE_BOUNDARY_SEMANTICS_VERSION
    information_set_sampler_version: str = DEFAULT_INFORMATION_SET_SAMPLER_VERSION
    sampler_rng_version: str = DEFAULT_SAMPLER_RNG_VERSION
    determinization_count: int = 1
    fallback_agent: str = DEFAULT_FALLBACK_AGENT
    fallback_agent_version: str = DEFAULT_FALLBACK_AGENT_VERSION
    temperature: float = 0.0
    selector_version: str = DEFAULT_SELECTOR_VERSION
    selector_rng_version: str = DEFAULT_SELECTOR_RNG_VERSION
    search_descriptor_id: str | None = PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    learned_turn_action_policy: str | None = DEFAULT_LEARNED_TURN_ACTION_POLICY
    search_continuation_policy: str | None = DEFAULT_SEARCH_CONTINUATION_POLICY
    schema_version: int = POLICY_DESCRIPTOR_SCHEMA_VERSION
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION,
            POLICY_DESCRIPTOR_SCHEMA_VERSION,
        }:
            raise PolicyCompatibilityError(
                f"unsupported policy descriptor schema {self.schema_version}"
            )
        if self.determinization_count < 1:
            raise ValueError("policy determinization count must be positive")
        if not isinstance(self.temperature, (float, int)) or isinstance(self.temperature, bool):
            raise ValueError("policy temperature must be numeric")
        temperature = float(self.temperature)
        if not torch.isfinite(torch.tensor(temperature)) or temperature < 0.0:
            raise ValueError("policy temperature must be finite and non-negative")
        object.__setattr__(self, "temperature", temperature)
        for name in (
            "checkpoint_id",
            "encoder_layout_fingerprint",
            "card_data_fingerprint",
            "effects_fingerprint",
            "engine_version",
            "rules_version",
            "information_policy_version",
            "afterstate_boundary_semantics_version",
            "information_set_sampler_version",
            "sampler_rng_version",
            "fallback_agent",
            "fallback_agent_version",
            "selector_version",
            "selector_rng_version",
        ):
            _require_string(getattr(self, name), f"policy descriptor {name}")

        if self.schema_version == LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            if any(
                value is not None
                for value in (
                    self.search_descriptor_id,
                    self.learned_turn_action_policy,
                    self.search_continuation_policy,
                )
            ):
                raise PolicyCompatibilityError(
                    "schema-v1 policy descriptors cannot contain schema-v2 fields"
                )
            if (
                self.fallback_agent != LEGACY_DEFAULT_FALLBACK_AGENT
                or self.fallback_agent_version != DEFAULT_FALLBACK_AGENT_VERSION
            ):
                raise PolicyCompatibilityError(
                    "schema-v1 policy descriptor fallback differs from legacy contract"
                )
        else:
            search_descriptor_id = _require_string(
                self.search_descriptor_id, "policy descriptor search_descriptor_id"
            )
            digest = search_descriptor_id.removeprefix("sha256:")
            if (
                not search_descriptor_id.startswith("sha256:")
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise PolicyCompatibilityError(
                    "policy descriptor search_descriptor_id must be a tagged sha256 digest"
                )
            _require_string(
                self.learned_turn_action_policy,
                "policy descriptor learned_turn_action_policy",
            )
            _require_string(
                self.search_continuation_policy,
                "policy descriptor search_continuation_policy",
            )
            if self.information_policy_version != PUBLIC_COVERED_INFORMATION_POLICY_VERSION:
                raise PolicyCompatibilityError(
                    "schema-v2 policy descriptor requires public-covered-v1 information"
                )

        object.__setattr__(
            self,
            "policy_id",
            _sha256_bytes(_canonical_json(self._identity_payload()).encode("utf-8")),
        )

    def _identity_payload(self) -> dict[str, object]:
        if self.schema_version == LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            names = (
                "checkpoint_id",
                "encoder_layout_fingerprint",
                "card_data_fingerprint",
                "effects_fingerprint",
                "engine_version",
                "rules_version",
                "information_policy_version",
                "action_schema_version",
                "decision_schema_version",
                "observation_schema_version",
                "value_position_schema_version",
                "public_boundary_schema_version",
                "afterstate_boundary_semantics_version",
                "information_set_sampler_version",
                "sampler_rng_version",
                "determinization_count",
                "fallback_agent",
                "fallback_agent_version",
                "temperature",
                "selector_version",
                "selector_rng_version",
                "schema_version",
            )
            return {name: getattr(self, name) for name in names}
        return cast(
            dict[str, object],
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "policy_id"
            },
        )

    def payload(self) -> dict[str, object]:
        """Return canonical JSON-compatible descriptor content and derived identity."""

        payload = self._identity_payload()
        payload["policy_id"] = self.policy_id
        return payload

    def dumps(self) -> str:
        """Serialize this immutable descriptor canonically."""

        return _canonical_json(self.payload())

    def save(self, path: str | Path) -> None:
        """Write a descriptor atomically; callers should name it by ``policy_id``."""

        _atomic_write_bytes(Path(path), _canonical_json_bytes(self.payload()))

    @classmethod
    def from_payload(cls, payload: object) -> PolicyDescriptor:
        """Decode an exact schema-v1 or schema-v2 descriptor and verify its identity."""

        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise PolicyCompatibilityError("policy descriptor must be a JSON object")
        schema_version = _require_int(payload.get("schema_version"), "schema_version", minimum=1)
        common_fields = {
            "checkpoint_id",
            "encoder_layout_fingerprint",
            "card_data_fingerprint",
            "effects_fingerprint",
            "engine_version",
            "rules_version",
            "information_policy_version",
            "action_schema_version",
            "decision_schema_version",
            "observation_schema_version",
            "value_position_schema_version",
            "public_boundary_schema_version",
            "afterstate_boundary_semantics_version",
            "information_set_sampler_version",
            "sampler_rng_version",
            "determinization_count",
            "fallback_agent",
            "fallback_agent_version",
            "temperature",
            "selector_version",
            "selector_rng_version",
            "schema_version",
            "policy_id",
        }
        if schema_version == LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            expected = common_fields
        elif schema_version == POLICY_DESCRIPTOR_SCHEMA_VERSION:
            expected = common_fields | {
                "search_descriptor_id",
                "learned_turn_action_policy",
                "search_continuation_policy",
            }
        else:
            raise PolicyCompatibilityError(f"unsupported policy descriptor schema {schema_version}")
        if set(payload) != expected:
            raise PolicyCompatibilityError("policy descriptor fields differ from schema")
        if schema_version == LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            search_descriptor_id = None
            learned_turn_action_policy = None
            search_continuation_policy = None
        else:
            search_descriptor_id = _require_string(
                payload["search_descriptor_id"], "search_descriptor_id"
            )
            learned_turn_action_policy = _require_string(
                payload["learned_turn_action_policy"], "learned_turn_action_policy"
            )
            search_continuation_policy = _require_string(
                payload["search_continuation_policy"], "search_continuation_policy"
            )
        raw_temperature = payload["temperature"]
        if isinstance(raw_temperature, bool) or not isinstance(raw_temperature, (int, float)):
            raise PolicyCompatibilityError("policy temperature must be numeric")
        descriptor = cls(
            checkpoint_id=_require_string(payload["checkpoint_id"], "checkpoint_id"),
            encoder_layout_fingerprint=_require_string(
                payload["encoder_layout_fingerprint"], "encoder_layout_fingerprint"
            ),
            card_data_fingerprint=_require_string(
                payload["card_data_fingerprint"], "card_data_fingerprint"
            ),
            effects_fingerprint=_require_string(
                payload["effects_fingerprint"], "effects_fingerprint"
            ),
            engine_version=_require_string(payload["engine_version"], "engine_version"),
            rules_version=_require_string(payload["rules_version"], "rules_version"),
            information_policy_version=_require_string(
                payload["information_policy_version"], "information_policy_version"
            ),
            action_schema_version=_require_int(
                payload["action_schema_version"], "action_schema_version"
            ),
            decision_schema_version=_require_int(
                payload["decision_schema_version"], "decision_schema_version"
            ),
            observation_schema_version=_require_int(
                payload["observation_schema_version"], "observation_schema_version"
            ),
            value_position_schema_version=_require_int(
                payload["value_position_schema_version"], "value_position_schema_version"
            ),
            public_boundary_schema_version=_require_int(
                payload["public_boundary_schema_version"], "public_boundary_schema_version"
            ),
            afterstate_boundary_semantics_version=_require_string(
                payload["afterstate_boundary_semantics_version"],
                "afterstate_boundary_semantics_version",
            ),
            information_set_sampler_version=_require_string(
                payload["information_set_sampler_version"], "information_set_sampler_version"
            ),
            sampler_rng_version=_require_string(
                payload["sampler_rng_version"], "sampler_rng_version"
            ),
            determinization_count=_require_int(
                payload["determinization_count"], "determinization_count", minimum=1
            ),
            fallback_agent=_require_string(payload["fallback_agent"], "fallback_agent"),
            fallback_agent_version=_require_string(
                payload["fallback_agent_version"], "fallback_agent_version"
            ),
            temperature=float(raw_temperature),
            selector_version=_require_string(payload["selector_version"], "selector_version"),
            selector_rng_version=_require_string(
                payload["selector_rng_version"], "selector_rng_version"
            ),
            search_descriptor_id=search_descriptor_id,
            learned_turn_action_policy=learned_turn_action_policy,
            search_continuation_policy=search_continuation_policy,
            schema_version=schema_version,
        )
        if _require_string(payload["policy_id"], "policy_id") != descriptor.policy_id:
            raise PolicyCompatibilityError("policy descriptor content-derived ID is invalid")
        return descriptor

    @classmethod
    def from_checkpoint(
        cls,
        manifest: CheckpointManifest,
        *,
        afterstate_boundary_semantics_version: str = DEFAULT_AFTERSTATE_BOUNDARY_SEMANTICS_VERSION,
        information_set_sampler_version: str = DEFAULT_INFORMATION_SET_SAMPLER_VERSION,
        sampler_rng_version: str = DEFAULT_SAMPLER_RNG_VERSION,
        determinization_count: int = 1,
        fallback_agent: str = DEFAULT_FALLBACK_AGENT,
        fallback_agent_version: str = DEFAULT_FALLBACK_AGENT_VERSION,
        temperature: float = 0.0,
        selector_version: str = DEFAULT_SELECTOR_VERSION,
        selector_rng_version: str = DEFAULT_SELECTOR_RNG_VERSION,
        search_descriptor_id: str = PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id,
    ) -> PolicyDescriptor:
        """Build a schema-v2 descriptor with fields copied from a compatible checkpoint."""

        return cls(
            checkpoint_id=manifest.checkpoint_id,
            encoder_layout_fingerprint=manifest.encoder_layout_fingerprint,
            card_data_fingerprint=manifest.card_data_fingerprint,
            effects_fingerprint=manifest.effects_fingerprint,
            engine_version=manifest.engine_version,
            rules_version=manifest.rules_version,
            information_policy_version=manifest.information_policy_version,
            action_schema_version=manifest.action_schema_version,
            decision_schema_version=manifest.decision_schema_version,
            observation_schema_version=manifest.observation_schema_version,
            value_position_schema_version=manifest.value_position_schema_version,
            public_boundary_schema_version=manifest.public_boundary_schema_version,
            afterstate_boundary_semantics_version=afterstate_boundary_semantics_version,
            information_set_sampler_version=information_set_sampler_version,
            sampler_rng_version=sampler_rng_version,
            determinization_count=determinization_count,
            fallback_agent=fallback_agent,
            fallback_agent_version=fallback_agent_version,
            temperature=temperature,
            selector_version=selector_version,
            selector_rng_version=selector_rng_version,
            search_descriptor_id=search_descriptor_id,
            learned_turn_action_policy=DEFAULT_LEARNED_TURN_ACTION_POLICY,
            search_continuation_policy=DEFAULT_SEARCH_CONTINUATION_POLICY,
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A validated model bundle loaded with CPU tensors only."""

    directory: Path
    manifest: CheckpointManifest
    model: ValueNetwork
    metrics: Mapping[str, object]
    optimizer_state: Mapping[str, object] | None


def _cpu_copy(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _validate_safe_state(value: object, name: str) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise CheckpointIntegrityError(f"{name} contains a non-CPU tensor")
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, dict):
        if not all(isinstance(key, (str, int)) for key in value):
            raise CheckpointIntegrityError(f"{name} has unsupported state-dict keys")
        for key, item in value.items():
            _validate_safe_state(item, f"{name}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_state(item, f"{name}[{index}]")
        return
    raise CheckpointIntegrityError(f"{name} contains unsupported state-dict data")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_torch(path: Path, state: Mapping[str, object]) -> None:
    torch.save(dict(state), path)
    with path.open("rb") as target:
        os.fsync(target.fileno())


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as target:
        target.write(_canonical_json_bytes(value))
        target.flush()
        os.fsync(target.fileno())


def _manifest_for(
    model: ValueNetwork,
    encoder_manifest: EncoderManifest,
    *,
    model_sha256: str,
    optimizer_sha256: str | None,
    metrics_sha256: str,
    training_dataset_ids: tuple[str, ...],
    parent_checkpoint_ids: tuple[str, ...],
    generation: int,
    optimizer_config: Mapping[str, object],
    creation_command: str,
) -> CheckpointManifest:
    return CheckpointManifest(
        architecture=VALUE_NETWORK_ARCHITECTURE,
        input_dimension=model.input_dimension,
        hidden_dimension=VALUE_NETWORK_HIDDEN_DIMENSION,
        encoder_version=encoder_manifest.encoder_version,
        encoder_layout_fingerprint=encoder_manifest.layout_fingerprint,
        card_data_fingerprint=encoder_manifest.card_data_fingerprint,
        rules_version=encoder_manifest.rules_version,
        information_policy_version=encoder_manifest.information_policy_version,
        engine_version=ENGINE_VERSION,
        effects_fingerprint=effects_fingerprint(),
        action_schema_version=ACTION_SCHEMA_VERSION,
        decision_schema_version=DECISION_SCHEMA_VERSION,
        observation_schema_version=encoder_manifest.observation_schema_version,
        state_schema_version=STATE_SCHEMA_VERSION,
        terminal_schema_version=TERMINAL_SCHEMA_VERSION,
        effect_runtime_schema_version=EFFECT_RUNTIME_SCHEMA_VERSION,
        value_position_schema_version=encoder_manifest.value_position_schema_version,
        public_boundary_schema_version=encoder_manifest.public_boundary_schema_version,
        training_dataset_ids=training_dataset_ids,
        parent_checkpoint_ids=parent_checkpoint_ids,
        generation=generation,
        optimizer_config_json=_canonical_json(
            _require_json_object(dict(optimizer_config), "optimizer_config")
        ),
        pytorch_version=torch.__version__,
        creation_command=creation_command,
        model_sha256=model_sha256,
        optimizer_sha256=optimizer_sha256,
        metrics_sha256=metrics_sha256,
    )


def save_checkpoint(
    checkpoint_root: str | Path,
    model: ValueNetwork,
    encoder_manifest: EncoderManifest,
    *,
    optimizer: Optimizer | None = None,
    metrics: Mapping[str, object] | None = None,
    training_dataset_ids: tuple[str, ...] = (),
    parent_checkpoint_ids: tuple[str, ...] = (),
    generation: int = 0,
    optimizer_config: Mapping[str, object] | None = None,
    creation_command: str = "",
) -> Path:
    """Atomically publish an immutable, CPU-only checkpoint under ``checkpoint_root``.

    The return value is ``checkpoint_root / <content-derived checkpoint-id>``. Existing
    bundle directories are never modified or reused.
    """

    if model.input_dimension != encoder_manifest.input_dimension:
        raise CheckpointCompatibilityError("model input dimension differs from encoder manifest")
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    metric_payload = _require_json_object(dict(metrics or {}), "metrics")
    config_payload = _require_json_object(dict(optimizer_config or {}), "optimizer_config")
    if not isinstance(creation_command, str):
        raise ValueError("creation_command must be a string")
    if generation < 0:
        raise ValueError("generation cannot be negative")
    if len(set(training_dataset_ids)) != len(training_dataset_ids):
        raise ValueError("training dataset IDs cannot repeat")
    if len(set(parent_checkpoint_ids)) != len(parent_checkpoint_ids):
        raise ValueError("parent checkpoint IDs cannot repeat")

    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-tmp-", dir=root))
    try:
        model_path = temporary / "model.pt"
        model_state = cast(Mapping[str, object], _cpu_copy(model.state_dict()))
        _validate_safe_state(dict(model_state), "model state")
        _write_torch(model_path, model_state)

        optimizer_digest: str | None = None
        if optimizer is not None:
            optimizer_path = temporary / "optimizer.pt"
            optimizer_state = cast(Mapping[str, object], _cpu_copy(optimizer.state_dict()))
            _validate_safe_state(dict(optimizer_state), "optimizer state")
            _write_torch(optimizer_path, optimizer_state)
            optimizer_digest = _sha256_file(optimizer_path)

        metrics_path = temporary / "metrics.json"
        _write_json(metrics_path, metric_payload)
        manifest = _manifest_for(
            model,
            encoder_manifest,
            model_sha256=_sha256_file(model_path),
            optimizer_sha256=optimizer_digest,
            metrics_sha256=_sha256_file(metrics_path),
            training_dataset_ids=training_dataset_ids,
            parent_checkpoint_ids=parent_checkpoint_ids,
            generation=generation,
            optimizer_config=config_payload,
            creation_command=creation_command,
        )
        target = root / manifest.checkpoint_id
        if target.exists():
            raise FileExistsError(f"immutable checkpoint already exists: {target}")
        _write_json(temporary / "manifest.json", manifest.payload())
        os.replace(temporary, target)
        # Best-effort durability for the directory rename.
        try:
            descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint_manifest(directory: str | Path) -> CheckpointManifest:
    """Load and validate only a checkpoint manifest, without deserializing tensors."""

    path = Path(directory) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"could not load checkpoint manifest: {error}") from error
    return CheckpointManifest.from_payload(payload)


def _verify_digest(path: Path, expected: str) -> None:
    if not path.is_file():
        raise CheckpointIntegrityError(f"checkpoint file is missing: {path.name}")
    actual = _sha256_file(path)
    if actual != expected:
        raise CheckpointIntegrityError(f"checkpoint digest mismatch for {path.name}")


def _safe_torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise CheckpointIntegrityError(
            "installed PyTorch does not support safe weights_only checkpoint loading"
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointIntegrityError(f"could not safely load {path.name}: {error}") from error


def _validate_manifest_compatibility(
    manifest: CheckpointManifest,
    encoder_manifest: EncoderManifest,
) -> None:
    expected = {
        "architecture": VALUE_NETWORK_ARCHITECTURE,
        "input_dimension": encoder_manifest.input_dimension,
        "hidden_dimension": VALUE_NETWORK_HIDDEN_DIMENSION,
        "encoder_version": encoder_manifest.encoder_version,
        "encoder_layout_fingerprint": encoder_manifest.layout_fingerprint,
        "card_data_fingerprint": encoder_manifest.card_data_fingerprint,
        "rules_version": RULES_VERSION,
        "information_policy_version": encoder_manifest.information_policy_version,
        "engine_version": ENGINE_VERSION,
        "effects_fingerprint": effects_fingerprint(),
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "terminal_schema_version": TERMINAL_SCHEMA_VERSION,
        "effect_runtime_schema_version": EFFECT_RUNTIME_SCHEMA_VERSION,
        "value_position_schema_version": encoder_manifest.value_position_schema_version,
        "public_boundary_schema_version": encoder_manifest.public_boundary_schema_version,
        "pytorch_version": torch.__version__,
    }
    for field_name, value in expected.items():
        if getattr(manifest, field_name) != value:
            raise CheckpointCompatibilityError(
                f"checkpoint {field_name} is incompatible with this runtime"
            )


def load_checkpoint(
    directory: str | Path,
    *,
    encoder_manifest: EncoderManifest | None = None,
    load_optimizer: bool = False,
) -> LoadedCheckpoint:
    """Verify and load a checkpoint with safe ``weights_only`` CPU deserialization."""

    root = Path(directory)
    manifest = load_checkpoint_manifest(root)
    if root.name != manifest.checkpoint_id:
        raise CheckpointIntegrityError("checkpoint directory name differs from manifest ID")
    current_encoder = encoder_manifest or build_encoder_manifest(
        information_policy_version=manifest.information_policy_version
    )
    _validate_manifest_compatibility(manifest, current_encoder)

    model_path = root / "model.pt"
    metrics_path = root / "metrics.json"
    _verify_digest(model_path, manifest.model_sha256)
    _verify_digest(metrics_path, manifest.metrics_sha256)
    raw_model = _safe_torch_load(model_path)
    if not isinstance(raw_model, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in raw_model.items()
    ):
        raise CheckpointIntegrityError("model.pt must contain only a tensor state dictionary")
    _validate_safe_state(raw_model, "model state")
    model = ValueNetwork(manifest.input_dimension)
    try:
        model.load_state_dict(cast(dict[str, Tensor], raw_model), strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError("checkpoint model state is incompatible") from error
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    try:
        metrics = _require_json_object(
            json.loads(metrics_path.read_text(encoding="utf-8")), "metrics"
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"could not load metrics.json: {error}") from error

    optimizer_state: Mapping[str, object] | None = None
    optimizer_path = root / "optimizer.pt"
    if manifest.optimizer_sha256 is None:
        if optimizer_path.exists():
            raise CheckpointIntegrityError("optimizer.pt exists but is absent from manifest")
    else:
        _verify_digest(optimizer_path, manifest.optimizer_sha256)
        if load_optimizer:
            raw_optimizer = _safe_torch_load(optimizer_path)
            if not isinstance(raw_optimizer, dict):
                raise CheckpointIntegrityError("optimizer.pt must contain a state dictionary")
            _validate_safe_state(raw_optimizer, "optimizer state")
            optimizer_state = cast(Mapping[str, object], raw_optimizer)

    return LoadedCheckpoint(root, manifest, model, metrics, optimizer_state)


def load_policy_descriptor(path: str | Path) -> PolicyDescriptor:
    """Load a canonical policy descriptor and verify its content-derived ID."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyCompatibilityError(f"could not load policy descriptor: {error}") from error
    return PolicyDescriptor.from_payload(payload)


def assert_policy_compatible(
    descriptor: PolicyDescriptor,
    manifest: CheckpointManifest,
) -> None:
    """Reject a policy that names weights or compatibility fields inconsistently."""

    expected = {
        "checkpoint_id": manifest.checkpoint_id,
        "encoder_layout_fingerprint": manifest.encoder_layout_fingerprint,
        "card_data_fingerprint": manifest.card_data_fingerprint,
        "effects_fingerprint": manifest.effects_fingerprint,
        "engine_version": manifest.engine_version,
        "rules_version": manifest.rules_version,
        "information_policy_version": manifest.information_policy_version,
        "action_schema_version": manifest.action_schema_version,
        "decision_schema_version": manifest.decision_schema_version,
        "observation_schema_version": manifest.observation_schema_version,
        "value_position_schema_version": manifest.value_position_schema_version,
        "public_boundary_schema_version": manifest.public_boundary_schema_version,
    }
    for field_name, value in expected.items():
        if getattr(descriptor, field_name) != value:
            raise PolicyCompatibilityError(f"policy {field_name} differs from checkpoint")


def make_cpu_evaluator(
    directory: str | Path,
    *,
    encoder_manifest: EncoderManifest | None = None,
) -> CpuBatchValueEvaluator:
    """Load a checkpoint and create its CPU evaluator without exposing optimizer state."""

    # Deferred import avoids a checkpoint/inference import cycle.
    from innovation_ai.training.inference import CpuBatchValueEvaluator

    loaded = load_checkpoint(directory, encoder_manifest=encoder_manifest)
    return CpuBatchValueEvaluator(loaded.model)
